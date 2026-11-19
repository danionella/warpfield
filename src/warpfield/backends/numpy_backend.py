"""NumPy/SciPy backend.

Runs on the CPU. It is the reference the GPU backends are tested against, and lets the shared code be
tested on machines without a GPU (e.g. the CI runners). It is never picked automatically.

All methods go through ``self.xp`` / ``self.ndi`` / ``self.windows``, so the CuPy backend can reuse them
by swapping in ``cupy`` / ``cupyx.scipy.ndimage`` / ``cupyx.scipy.signal.windows``.
"""

import numpy as np
import scipy.ndimage
import scipy.signal.windows

from .array_backend import ArrayBackend


class NumpyBackend(ArrayBackend):
    name = "numpy"

    def __init__(self):
        self.xp = np
        self.ndi = scipy.ndimage
        self.windows = scipy.signal.windows

    # -------------------------------------------------------------- conversion
    def is_array(self, x):
        return isinstance(x, np.ndarray)

    def asarray(self, x, dtype=None):
        return np.ascontiguousarray(self._host_if_foreign(x), dtype=self.dtype(dtype))

    def to_numpy(self, x):
        return np.asarray(self._host_if_foreign(x))

    def copy(self, x):
        return x.copy()

    # ----------------------------------------------------- creation / indexing
    def indices(self, shape, dtype=None):
        return self.xp.indices(tuple(int(s) for s in shape), dtype=self.dtype(dtype) or self.int32)

    # ------------------------------------------------------------------ ndimage
    def gaussian_filter(self, x, sigma, mode="reflect", truncate=4.0):
        return self.ndi.gaussian_filter(x, sigma, mode=mode, truncate=truncate)

    def convolve_leading(self, x, kernel, mode="constant"):
        kernel = self.asarray(kernel, x.dtype)
        kernel = kernel.reshape(kernel.shape + (1,) * (x.ndim - kernel.ndim))
        return self.ndi.convolve(x, kernel, mode=mode)

    def median_filter(self, x, size, mode="nearest"):
        return self.ndi.median_filter(x, size=size, mode=mode)

    def map_coordinates(self, x, coords, order=1, mode="nearest", cval=0.0):
        return self.ndi.map_coordinates(x, coords, order=order, mode=mode, cval=cval)

    def zoom(self, x, factors, order=1):
        return self.ndi.zoom(x, factors, order=order)

    def tukey(self, n, alpha):
        return self.asarray(self.windows.tukey(int(n), float(alpha)), self.float32)

    def sliding_block(self, x, block_size, block_stride):
        block_size = np.asarray(block_size, dtype="int64") * np.ones(x.ndim, dtype="int64")
        block_stride = np.asarray(block_stride, dtype="int64") * np.ones(x.ndim, dtype="int64")
        shape = np.r_[1 + (np.array(x.shape) - block_size) // block_stride, block_size]
        strides = np.r_[block_stride * np.array(x.strides), x.strides]
        return self.xp.lib.stride_tricks.as_strided(x, tuple(int(s) for s in shape), tuple(int(s) for s in strides))

    # ------------------------------------------------------------- warp kernel
    def warp_volume(self, vol, disp_field, disp_scale, disp_offset, out=None):
        """NumPy port of the CUDA warp kernel (``_cupy/warp.py``). Slow; meant for tests and small data."""
        f32 = np.float32
        vol = self.asarray(vol, f32)
        disp_field = self.asarray(disp_field, f32)
        disp_scale = self.asarray(disp_scale, f32).ravel()
        disp_offset = self.asarray(disp_offset, f32).ravel()
        if out is None:
            out = np.zeros(vol.shape, dtype=f32)
        assert out.dtype == f32, "out must be float32"
        j, k = np.indices(out.shape[1:], dtype=f32)
        for i in range(out.shape[0]):  # one slab at a time to bound memory
            ii = np.full(j.shape, i, dtype=f32)
            x, y, z = ii / disp_scale[0] + disp_offset[0], j / disp_scale[1] + disp_offset[1], k / disp_scale[2] + disp_offset[2]
            d = [_trilinear(disp_field[c], x, y, z, extrapolate="linear") for c in range(3)]
            out[i] = _trilinear(vol, ii + d[0], j + d[1], k + d[2], extrapolate="zero")
        return out


def _trilinear(arr, x, y, z, extrapolate):
    """Vectorised port of ``trilinear_interp`` in the CUDA kernel.

    Coordinates are clipped to [-1, shape]. Outside ``arr``, values are 0 (``"zero"``) or extrapolated
    linearly from the two nearest voxels (``"linear"``).
    """
    shape = arr.shape
    coords = [np.clip(c, np.float32(-1), np.float32(n)) for c, n in zip((x, y, z), shape)]
    lo = [np.floor(c).astype(np.int64) for c in coords]
    frac = [(c - l).astype(np.float32) for c, l in zip(coords, lo)]

    def fetch(ix):
        inside = np.ones(ix[0].shape, dtype=bool)
        clamped = []
        for i, n in zip(ix, shape):
            inside &= (i >= 0) & (i < n)
            clamped.append(np.clip(i, 0, n - 1))
        v = arr[tuple(clamped)]
        if extrapolate == "zero":
            return np.where(inside, v, np.float32(0))
        # one voxel further inside, for linear extrapolation
        inner = [np.clip(np.where(i < 0, c + 1, np.where(i >= n, c - 1, c)), 0, n - 1) for i, c, n in zip(ix, clamped, shape)]
        return np.where(inside, v, 2 * v - arr[tuple(inner)])

    (x0, y0, z0), (xd, yd, zd) = lo, frac
    c00 = fetch((x0, y0, z0)) * (1 - xd) + fetch((x0 + 1, y0, z0)) * xd
    c01 = fetch((x0, y0, z0 + 1)) * (1 - xd) + fetch((x0 + 1, y0, z0 + 1)) * xd
    c10 = fetch((x0, y0 + 1, z0)) * (1 - xd) + fetch((x0 + 1, y0 + 1, z0)) * xd
    c11 = fetch((x0, y0 + 1, z0 + 1)) * (1 - xd) + fetch((x0 + 1, y0 + 1, z0 + 1)) * xd
    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd
    return c0 * (1 - zd) + c1 * zd
