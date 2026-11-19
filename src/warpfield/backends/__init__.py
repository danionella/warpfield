"""Array backends: NumPy (CPU), CuPy (NVIDIA GPUs) and MLX (Apple Silicon GPUs). See ``array_backend.py``."""

from .array_backend import COMMON_XP, ArrayBackend
from .selection import (
    available_array_backends,
    backend_of,
    get_array_backend,
    like_input,
    resolve_backend,
    set_default_array_backend,
)

__all__ = [
    "ArrayBackend",
    "COMMON_XP",
    "available_array_backends",
    "backend_of",
    "get_array_backend",
    "like_input",
    "resolve_backend",
    "set_default_array_backend",
]
