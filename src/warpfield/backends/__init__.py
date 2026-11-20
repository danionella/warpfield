"""Array backends: NumPy (CPU), CuPy (NVIDIA GPUs) and MLX (Apple Silicon GPUs). See ``array_backend.py``."""

from .array_backend import COMMON_XP, ArrayBackend
from .selection import (
    available_array_backends,
    backend_of,
    current_backend,
    get_array_backend,
    like_input,
    set_array_backend,
)

__all__ = [
    "ArrayBackend",
    "COMMON_XP",
    "available_array_backends",
    "backend_of",
    "current_backend",
    "get_array_backend",
    "like_input",
    "set_array_backend",
]
