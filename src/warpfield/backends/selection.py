"""Choosing the array backend.

warpfield runs on one backend per process (``current_backend``): the one set with ``set_array_backend``, else
the first usable GPU backend (CuPy, then MLX). The NumPy backend runs on the CPU and is only used when set by
name, so registration never silently falls back to a slow CPU path.
"""

from functools import lru_cache
from importlib.util import find_spec

from .array_backend import ArrayBackend

_GPU_PREFERENCE = ("cupy", "mlx")
_instances = {}
_default = None


_unusable_reason = {}  # backend name -> why it is installed but not usable (shown in error messages)


@lru_cache(maxsize=None)
def _cupy_usable():
    if find_spec("cupy") is None:
        return False
    try:
        import cupy

        if cupy.cuda.runtime.getDeviceCount() < 1:
            _unusable_reason["cupy"] = "no CUDA GPU found"
            return False
        # A CuPy wheel without the CUDA libraries imports fine and only fails when a library is first used, so
        # use the ones warpfield needs: the modules the backend imports, NVRTC (kernels), cuBLAS and cuFFT.
        import cupyx.scipy.ndimage
        import cupyx.scipy.signal.windows

        x = cupy.ones((8, 8), dtype=cupy.float32)
        float((x @ x).sum() + cupy.fft.rfft2(x).real.sum())
        return True
    except Exception as e:
        _unusable_reason["cupy"] = f"{type(e).__name__}: {e}"
        return False


def _mlx_usable():  # not cached: depends on mx.default_device(), which users can change
    if find_spec("mlx") is None:
        return False
    try:
        import mlx.core as mx

        # The MLX warp kernel is Metal-only so far, so MLX on CUDA does not count yet
        return mx.default_device().type == mx.DeviceType.gpu and mx.metal.is_available()
    except Exception:
        return False


_USABLE = {"cupy": _cupy_usable, "mlx": _mlx_usable, "numpy": lambda: True}


def _create(name):
    if name == "numpy":
        from .numpy_backend import NumpyBackend

        return NumpyBackend()
    if name == "cupy":
        from .cupy_backend import CupyBackend

        return CupyBackend()
    if name == "mlx":
        from .mlx_backend import MlxBackend

        return MlxBackend()
    raise ValueError(f"Unknown backend '{name}'. Known backends: {list(_USABLE)}")


def available_array_backends():
    """Names of the backends that can run on this machine, GPU backends first."""
    return [name for name in (*_GPU_PREFERENCE, "numpy") if _USABLE[name]()]


def set_array_backend(name):
    """Set the backend warpfield runs on: "cupy", "mlx" or "numpy" (CPU, slow). "auto" or None restores the
    automatic choice (the first available GPU backend).

    Everything computes on this backend from then on. Existing WarpMaps follow the switch (their arrays move to the
    new backend when next used); your own arrays keep their type, and results are returned in it. A
    RegistrationPyramid is tied to the backend it was created on, so create a new one after switching.
    """
    global _default
    if name in (None, "auto", "default"):
        _default = None
    else:
        _default = get_array_backend(name).name  # fails early if not usable; normalises e.g. "MLX" or objects


def get_array_backend(backend="auto") -> ArrayBackend:
    """Return a backend object.

    Args:
        backend (str or ArrayBackend): "auto"/"default"/None, a backend name ("cupy", "mlx", "numpy"),
            or a backend object (returned unchanged).
    """
    if isinstance(backend, ArrayBackend):
        return backend
    name = backend.lower() if isinstance(backend, str) else backend
    if name in (None, "auto", "default"):
        name = _default or next((n for n in _GPU_PREFERENCE if _USABLE[n]()), None)
        if name is None:
            raise RuntimeError(
                "No GPU backend found. warpfield needs CuPy with a CUDA GPU (pip install 'warpfield[cuda12]' or "
                "'warpfield[cuda13]', matching your CUDA driver), or MLX on Apple Silicon. "
                "Call warpfield.set_backend('numpy') to run on the CPU (slow)." + _reasons()
            )
    if name in _USABLE and not _USABLE[name]():
        raise RuntimeError(f"Backend '{name}' is not usable on this machine (missing package or GPU)." + _reasons())
    return _instance(name)


def _reasons():
    return "".join(f"\n{name} is installed but not usable: {why}" for name, why in _unusable_reason.items())


def _instance(name):
    if name not in _instances:
        _instances[name] = _create(name)
    return _instances[name]


def backend_of(x) -> ArrayBackend:
    """Return the backend an array belongs to (NumPy for anything that is not a CuPy, MLX or CUDA array).

    Other CUDA arrays (PyTorch CUDA tensors, Numba, JAX, ...: anything with ``__cuda_array_interface__``) belong
    to CuPy, which uses them without a copy, so results stay on the GPU (as CuPy arrays).

    No usability check for CuPy and MLX arrays: an array that exists can always be converted, e.g. an MLX array
    when MLX runs on the CPU or on CUDA, where the MLX backend cannot register (its warp kernel needs Metal).
    """
    module = type(x).__module__.split(".")[0]
    if module not in ("cupy", "mlx") and hasattr(x, "__cuda_array_interface__") and _cupy_usable():
        module = "cupy"
    return _instance(module if module in ("cupy", "mlx") else "numpy")


def current_backend() -> ArrayBackend:
    """The backend warpfield runs on: the one set with ``set_array_backend``, else the first usable GPU backend."""
    return get_array_backend("auto")


def like_input(out, inp):
    """Return ``out`` on the backend of ``inp``, e.g. NumPy if the user passed a NumPy array."""
    B = backend_of(inp)
    return out if B.is_array(out) else B.asarray(out)
