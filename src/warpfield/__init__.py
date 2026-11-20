"""
.. include:: ../../README.md
"""

from . import backends, ndimage, register, utils, warp
from .register import register_volumes, Recipe, WarpMap
from .utils import load_data, set_default_backend, get_available_backends

__all__ = [
    "register_volumes",
    "Recipe",
    "WarpMap",
    "load_data",
    "get_available_backends",
    "set_default_backend",
    # submodules: listed so that pdoc builds their documentation pages (linked from the README)
    "register",
    "ndimage",
    "warp",
    "utils",
    "backends",
]
