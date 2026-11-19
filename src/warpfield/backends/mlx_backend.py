"""MLX backend (Apple Silicon GPUs; MLX also runs on CUDA, but the warp kernel is Metal-only so far).

Wraps the MLX implementations in ``_mlx/``. Cases those do not cover run on the CPU through the NumPy backend,
which is cheap on Apple Silicon because CPU and GPU share memory.
"""

import numpy as np

from .array_backend import ArrayBackend


class MlxBackend(ArrayBackend):
    name = "mlx"

    def __init__(self):
        import mlx.core as mx

        from .selection import get_array_backend

        self.xp = mx
        self._cpu = get_array_backend("numpy")

    @property
    def float_hp(self):
        return self.xp.float32  # MLX GPUs have no float64

    def _on_cpu(self, method, x, *args, **kwargs):
        """Run a NumPy backend method on ``x`` and return the result as an MLX array."""
        return self.asarray(getattr(self._cpu, method)(self.to_numpy(x), *args, **kwargs))

    # -------------------------------------------------------------- conversion
    def is_array(self, x):
        return isinstance(x, self.xp.array)

    def asarray(self, x, dtype=None):
        dtype = self.dtype(dtype)
        if not self.is_array(x):
            x = np.asarray(self._host_if_foreign(x))
            if dtype is None and x.dtype == np.float64:
                dtype = self.float32  # float64 does not run on the GPU
            x = self.xp.array(x)
        if dtype is not None and x.dtype != dtype:
            x = x.astype(dtype)
        return x

    def to_numpy(self, x):
        return np.asarray(self._host_if_foreign(x))

    def copy(self, x):
        return self.xp.array(x)

    # ---------------------------------------------------------- linear algebra
    # MLX runs these on the CPU only; with shared memory that costs no transfer.
    def lstsq(self, a, b):
        # NumPy in float64 (SVD), which also handles rank-deficient systems, e.g. an affine fit with a single
        # block along one axis. MLX has no lstsq, and the normal equations are singular in that case.
        a, b = (np.asarray(self.to_numpy(v), dtype=np.float64) for v in (a, b))
        return self.asarray(np.linalg.lstsq(a, b, rcond=None)[0], self.float32)

    def det(self, a):
        return self.xp.linalg.det(a, stream=self.xp.cpu)

    def gradient(self, f, *spacing, edge_order=1):
        return [self.asarray(g) for g in self._cpu.gradient(self.to_numpy(f), *spacing, edge_order=edge_order)]

    # ------------------------------------------------------------------ ndimage
    def gaussian_filter(self, x, sigma, mode="reflect", truncate=4.0):
        from ._mlx.ndimage import gaussian_filter

        if mode in ("reflect", "constant"):
            return gaussian_filter(x, sigma, mode=mode, truncate=truncate)
        return self._on_cpu("gaussian_filter", x, sigma, mode=mode, truncate=truncate)

    def convolve_leading(self, x, kernel, mode="constant"):
        from ._mlx.ndimage import convolve_2d_leading

        if np.ndim(kernel) == 2 and mode == "constant":
            return convolve_2d_leading(x, self.asarray(kernel, self.float32))
        return self._on_cpu("convolve_leading", x, self.to_numpy(kernel), mode=mode)

    def median_filter(self, x, size, mode="nearest"):
        from ._mlx.ndimage import median_filter_1333

        if mode == "nearest" and np.broadcast_to(size, x.ndim).tolist() == [1, 3, 3, 3]:
            return median_filter_1333(x)
        return self._on_cpu("median_filter", x, size, mode=mode)

    def map_coordinates(self, x, coords, order=1, mode="nearest", cval=0.0):
        from ._mlx.ndimage import map_coordinates

        if order == 1 and mode == "nearest" and x.ndim == 3:
            return map_coordinates(x, self.asarray(coords, self.float32))
        return self._on_cpu("map_coordinates", x, self.to_numpy(coords), order=order, mode=mode, cval=cval)

    def zoom(self, x, factors, order=1):
        return self._on_cpu("zoom", x, factors, order=order)

    def tukey(self, n, alpha):
        return self.asarray(self._cpu.tukey(n, alpha))

    def sliding_block(self, x, block_size, block_stride):
        from ._mlx.ndimage import sliding_block

        return sliding_block(x, block_size, block_stride)

    # ------------------------------------------------------------- warp kernel
    def warp_volume(self, vol, disp_field, disp_scale, disp_offset, out=None):
        if not self.xp.metal.is_available():
            raise NotImplementedError("The MLX warp kernel needs Metal; a CUDA version is not written yet.")
        from ._mlx.warp import warp_volume_mlx

        f32 = self.float32
        res = warp_volume_mlx(
            self.asarray(vol, f32),
            self.asarray(disp_field, f32),
            self.asarray(disp_scale, f32),
            self.asarray(disp_offset, f32),
            out_shape=None if out is None else out.shape,
        )
        if out is None:
            return res
        assert out.dtype == f32, "out must be float32"
        out[:] = res  # MLX kernels cannot write in place; copying keeps the `out` contract (measured: no extra cost)
        return out

    # ---------------------------------------------------- evaluation / memory
    def eval(self, *arrays):
        self.xp.eval(*arrays)

    def end_step(self, *arrays):
        self.xp.eval(*arrays)
        self.xp.clear_cache()  # without this, MLX's buffer cache grows over the registration

    def clear_cache(self):
        self.xp.clear_cache()
