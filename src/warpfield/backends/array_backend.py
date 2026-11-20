"""Thin layer over the array libraries that warpfield runs on (NumPy, CuPy, MLX).

Shared code (WarpMap, WarpMapper, RegistrationPyramid, ndimage helpers) is written once, against a
backend object ``B``:

- ``B.xp`` is the array module (``numpy``, ``cupy`` or ``mlx.core``). Use it for the functions listed
  in ``COMMON_XP``, which exist with the same NumPy-style signature in all three libraries
  (checked by ``tests/test_array_backends.py``).
- Everything else goes through a ``B.<method>`` defined on ``ArrayBackend``: conversion, filters,
  interpolation, FFTs, linear algebra, the warp kernel and memory handling.

Rules for shared code:

- dtypes are ``B.float32`` / ``B.complex64`` (MLX does not accept dtype strings)
- no ``.get()``, ``.copy()``, ``np.asarray(x)`` or ``order=`` on backend arrays: use ``B.to_numpy``,
  ``B.copy`` and ``B.asarray``
- no float64 on the device (MLX GPUs lack it); ``B.float_hp`` is the most precise float available
- call ``B.end_step(...)`` after each registration step, and ``B.eval(...)`` where intermediates should be
  computed early to bound memory; on lazy backends (MLX) they force computation, elsewhere they do nothing

Where code runs. warpfield runs on one backend per process (``current_backend()``: the one set with
``warpfield.set_backend``, else the first usable GPU backend). Public functions and classes (``register_volumes``,
``WarpMap``, the ``ndimage`` functions, the ``Recipe`` pieces ``RegFilter``, ``Projector``, ``Smoother``, ...)
compute on it and return results in the type of their input (``like_input``), so NumPy in gives NumPy out. A
``WarpMap`` also uses the current backend: after a switch, its arrays are converted the first time they are used.

Inside the pipeline (``WarpMapper``, ``RegistrationPyramid``), everything stays on ``B``: use ``B.*``
methods (and private helpers such as ``WarpMap._warp``) rather than entry points, and pass whatever user
code returns through ``B.asarray``, since a custom callable may return e.g. a NumPy array.
"""

from __future__ import annotations

import numpy as np

# Functions shared code may call as ``B.xp.<name>``: the ones it uses, all with the same NumPy-style signature
# in numpy, cupy and mlx.core. Add a name when shared code needs it (the test checks every backend has it).
COMMON_XP = (
    "arange", "argmax", "broadcast_to", "concatenate", "conj", "cos", "einsum", "exp", "eye", "maximum",
    "ones", "real", "reshape", "sqrt", "stack", "sum", "take", "zeros", "zeros_like",
    "float32", "complex64", "int32",
    "fft.rfftn", "fft.irfftn", "fft.fftshift", "fft.fftfreq",
)  # fmt: skip


class ArrayBackend:
    """Interface of an array backend, with default implementations that only use ``COMMON_XP``.

    Subclasses set ``name`` and ``xp`` and override what their library does differently.
    """

    name: str = "base"
    xp = None  # array module

    def __repr__(self):
        return f"{type(self).__name__}(name={self.name!r})"

    def __reduce__(self):
        # Backends hold modules, which cannot be pickled; unpickle to this process's backend of the same name
        # (so WarpMaps can be pickled, deep-copied and sent to worker processes). No usability check, as in
        # backend_of: restoring an object should not fail where creating it did not.
        from .selection import _instance

        return (_instance, (self.name,))

    # ------------------------------------------------------------------ dtypes
    @property
    def float32(self):
        return self.xp.float32

    @property
    def complex64(self):
        return self.xp.complex64

    @property
    def int32(self):
        return self.xp.int32

    @property
    def float_hp(self):
        """Most precise float type that runs on this backend's device."""
        return self.xp.float64

    def dtype(self, dt):
        """Turn a dtype given as string, NumPy dtype or backend dtype into a backend dtype."""
        if dt is None:
            return None
        if isinstance(dt, str):
            return getattr(self.xp, dt)
        try:
            return getattr(self.xp, np.dtype(dt).name)
        except TypeError:  # already a backend dtype that NumPy does not understand (MLX)
            return dt

    # -------------------------------------------------------------- conversion
    def is_array(self, x) -> bool:
        """True if ``x`` is an array of this backend."""
        raise NotImplementedError

    def asarray(self, x, dtype=None):
        """Return ``x`` as a (C-contiguous) array of this backend. Avoids copies where possible.

        Accepts NumPy arrays, arrays of other backends, lists and scalars.
        """
        raise NotImplementedError

    def to_numpy(self, x) -> np.ndarray:
        """Return ``x`` as a NumPy array (copies device arrays to the host)."""
        raise NotImplementedError

    def _host_if_foreign(self, x):
        """``x`` copied to the host if it is an array of another GPU backend; anything else unchanged.

        Each library can take its own arrays, NumPy arrays, lists and scalars; arrays of another GPU library
        have to go through NumPy.
        """
        from .selection import backend_of

        other = backend_of(x)
        return x if other.name in ("numpy", self.name) else other.to_numpy(x)

    def copy(self, x):
        """Return a copy of a backend array."""
        raise NotImplementedError

    # ----------------------------------------------------- creation / indexing
    def indices(self, shape, dtype=None):
        """Like ``numpy.indices``."""
        xp = self.xp
        dtype = self.dtype(dtype) or self.int32
        shape = tuple(int(s) for s in shape)
        grids = []
        for d, n in enumerate(shape):
            s = [1] * len(shape)
            s[d] = n
            grids.append(xp.broadcast_to(xp.reshape(xp.arange(n, dtype=dtype), s), shape))
        return xp.stack(grids)

    def argmax_2d(self, x):
        """Position of the maximum over the last two axes.

        Replaces ``unravel_index(argmax(x, axis=(-2, -1)), x.shape[-2:])``, which MLX lacks.

        Returns:
            tuple: (row, column) signed int32 arrays of shape ``x.shape[:-2]``
        """
        n = x.shape[-1]
        flat = self.xp.argmax(self.xp.reshape(x, (*x.shape[:-2], -1)), axis=-1)
        flat = flat.astype(self.int32)  # MLX returns unsigned ints, which wrap when offsets are subtracted
        return flat // n, flat % n

    # ---------------------------------------------------------- linear algebra
    def lstsq(self, a, b):
        """Least-squares solution of ``a @ x = b``."""
        return self.xp.linalg.lstsq(a, b, rcond=None)[0]

    def det(self, a):
        """Determinant over the last two axes."""
        return self.xp.linalg.det(a)

    def gradient(self, f, *spacing, edge_order=1):
        """Like ``numpy.gradient``; ``spacing`` are scalars, one per axis."""
        return self.xp.gradient(f, *[float(s) for s in spacing], edge_order=edge_order)

    # ------------------------------------------------------------------ ndimage
    def gaussian_filter(self, x, sigma, mode="reflect", truncate=4.0):
        """Like ``scipy.ndimage.gaussian_filter``. A sigma of 0 leaves that axis unfiltered.

        Only ``mode="reflect"`` and ``mode="constant"`` (with cval=0) are needed and guaranteed.
        """
        raise NotImplementedError

    def convolve_leading(self, x, kernel, mode="constant"):
        """Convolve ``x`` with ``kernel`` over the first ``kernel.ndim`` axes of ``x``.

        Same result as ``scipy.ndimage.convolve(x, kernel[..., None, ...], mode=mode)``, with the
        kernel padded by singleton axes.
        """
        raise NotImplementedError

    def median_filter(self, x, size, mode="nearest"):
        """Like ``scipy.ndimage.median_filter``."""
        raise NotImplementedError

    def map_coordinates(self, x, coords, order=1, mode="nearest", cval=0.0):
        """Like ``scipy.ndimage.map_coordinates``. ``coords`` has shape (x.ndim, ...)."""
        raise NotImplementedError

    def zoom(self, x, factors, order=1):
        """Like ``scipy.ndimage.zoom``."""
        raise NotImplementedError

    def tukey(self, n, alpha):
        """Tukey window of length ``n`` as a float32 backend array."""
        raise NotImplementedError

    def sliding_block(self, x, block_size, block_stride):
        """View of a C-contiguous array as blocks: shape ``(*n_blocks, *block_size)``.

        Libraries count strides differently (bytes vs. elements), so this lives in the backend.
        """
        raise NotImplementedError

    # --------------------------------------------------------------------- FFT
    def rfft2(self, x):
        """Real FFT over the last two axes."""
        return self.xp.fft.rfftn(x, axes=(-2, -1))

    def irfft2(self, X, s=None):
        """Inverse real FFT over the last two axes.

        Args:
            s (tuple, optional): output size of the last two axes. If None, the last axis has size
                ``2 * (X.shape[-1] - 1)``, as in ``numpy.fft.irfftn``.
        """
        return self.xp.fft.irfftn(X, s=s, axes=(-2, -1))

    # ------------------------------------------------------------- warp kernel
    def warp_volume(self, vol, disp_field, disp_scale, disp_offset, out=None):
        """Warp a 3D volume with a (coarse) displacement field.

        For each output voxel ``p``, the displacement ``d`` is interpolated from ``disp_field`` at
        ``p / disp_scale + disp_offset`` (linear extrapolation outside the field), and the output is
        ``vol`` interpolated at ``p + d`` (zero outside ``vol``).

        Args:
            vol: 3D volume
            disp_field: displacement field of shape (3, *field_shape)
            disp_scale, disp_offset: 3-element arrays
            out: optional float32 output array; defines the output shape (default: ``vol.shape``)

        Returns:
            float32 backend array (``out`` if it was given)
        """
        raise NotImplementedError

    # ---------------------------------------------------- evaluation / memory
    def eval(self, *arrays):
        """Force computation of lazy arrays. Does nothing on eager backends."""
        return None

    def end_step(self, *arrays):
        """Called after each registration step with the arrays that are kept.

        Lazy backends compute them here and release intermediate buffers.
        """
        self.eval(*arrays)

    def clear_cache(self):
        """Free cached device memory (e.g. FFT plans or buffer caches)."""
        return None
