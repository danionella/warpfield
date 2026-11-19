"""Choosing an array backend.

``get_array_backend("auto")`` returns the first usable GPU backend (CuPy). The NumPy backend runs
on the CPU and is only used when asked for by name, so registration never silently falls back to a
slow CPU path.
"""

from importlib.util import find_spec

from .array_backend import ArrayBackend

_GPU_PREFERENCE = ("cupy",)
_instances = {}
_default = None


def _cupy_usable():
    if find_spec("cupy") is None:
        return False
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


_USABLE = {"cupy": _cupy_usable, "numpy": lambda: True}


def _create(name):
    if name == "numpy":
        from .numpy_backend import NumpyBackend

        return NumpyBackend()
    if name == "cupy":
        from .cupy_backend import CupyBackend

        return CupyBackend()
    raise ValueError(f"Unknown backend '{name}'. Known backends: {list(_USABLE)}")


def available_array_backends():
    """Names of the backends that can run on this machine, GPU backends first."""
    return [name for name in (*_GPU_PREFERENCE, "numpy") if _USABLE[name]()]


def set_default_array_backend(name):
    """Set the backend that ``"auto"`` resolves to ("cupy" or "numpy"). "auto" or None restores automatic
    selection."""
    global _default
    if name in (None, "auto", "default"):
        _default = None
    else:
        _default = get_array_backend(name).name  # fails early if not usable; normalises e.g. "MLX" or objects


def get_array_backend(backend="auto") -> ArrayBackend:
    """Return a backend object.

    Args:
        backend (str or ArrayBackend): "auto"/"default"/None, a backend name ("cupy", "numpy"),
            or a backend object (returned unchanged).
    """
    if isinstance(backend, ArrayBackend):
        return backend
    name = backend.lower() if isinstance(backend, str) else backend
    if name in (None, "auto", "default"):
        name = _default or next((n for n in _GPU_PREFERENCE if _USABLE[n]()), None)
        if name is None:
            raise RuntimeError(
                "No GPU backend found. warpfield needs CuPy with a CUDA GPU. "
                "Pass backend='numpy' to run on the CPU (slow)."
            )
    if name not in _instances:
        if name in _USABLE and not _USABLE[name]():
            raise RuntimeError(f"Backend '{name}' is not usable on this machine (missing package or GPU).")
        _instances[name] = _create(name)
    return _instances[name]


def backend_of(x) -> ArrayBackend:
    """Return the backend an array belongs to (NumPy for anything that is not a CuPy array)."""
    module = type(x).__module__.split(".")[0]
    if module == "cupy":
        return get_array_backend(module)
    return get_array_backend("numpy")


def resolve_backend(x, backend=None) -> ArrayBackend:
    """Backend to compute ``x`` on.

    A backend given by name (or object) wins. Otherwise a CuPy array stays on its own backend, and
    anything else (NumPy arrays, lists) goes to the default GPU backend.
    """
    if backend not in (None, "auto", "default"):
        return get_array_backend(backend)
    B = backend_of(x)
    return B if B.name != "numpy" else get_array_backend("auto")


def like_input(out, inp):
    """Return ``out`` on the backend of ``inp``, e.g. NumPy if the user passed a NumPy array."""
    B = backend_of(inp)
    return out if B.is_array(out) else B.asarray(out)
