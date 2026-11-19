"""CuPy backend (NVIDIA GPUs).

CuPy and ``cupyx.scipy`` mirror NumPy and SciPy, so this backend reuses the NumPy backend with the
modules swapped. It only adds device transfers and the CUDA warp kernel. FFT plans are cached by CuPy's own
(size-limited) plan cache.
"""

import numpy as np

from .numpy_backend import NumpyBackend


class CupyBackend(NumpyBackend):
    name = "cupy"

    def __init__(self):
        import cupy
        import cupyx.scipy.ndimage
        import cupyx.scipy.signal.windows

        self.xp = cupy
        self.ndi = cupyx.scipy.ndimage
        self.windows = cupyx.scipy.signal.windows

    # -------------------------------------------------------------- conversion
    def is_array(self, x):
        return isinstance(x, self.xp.ndarray)

    def asarray(self, x, dtype=None):
        return self.xp.asarray(self._host_if_foreign(x), dtype=self.dtype(dtype), order="C")

    def to_numpy(self, x):
        return x.get() if self.is_array(x) else np.asarray(self._host_if_foreign(x))

    # ------------------------------------------------------------- warp kernel
    def warp_volume(self, vol, disp_field, disp_scale, disp_offset, out=None):
        from ._cupy.warp import warp_volume_cupy

        f32 = self.float32
        return warp_volume_cupy(
            self.asarray(vol, f32),
            self.asarray(disp_field, f32),
            self.asarray(disp_scale, f32),
            self.asarray(disp_offset, f32),
            out=out,
        )

    # ---------------------------------------------------- evaluation / memory
    def clear_cache(self):
        self.xp.fft.config.get_plan_cache().clear()
