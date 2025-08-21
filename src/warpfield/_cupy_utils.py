"""
Utilities for conditional CuPy imports and GPU availability detection.

This module provides a centralized way to handle CuPy imports and GPU availability
checking, allowing the package to function even when CUDA libraries are not available.
"""

import warnings
from typing import Any, Optional

# Global variables to track CuPy availability
_CUPY_AVAILABLE = None
_GPU_AVAILABLE = None
_CUPY_MODULE = None
_CUPYX_MODULE = None


def _check_cupy_available() -> bool:
    """Check if CuPy can be imported and CUDA libraries are available."""
    global _CUPY_AVAILABLE, _CUPY_MODULE, _CUPYX_MODULE
    
    if _CUPY_AVAILABLE is not None:
        return _CUPY_AVAILABLE
    
    try:
        import cupy as cp
        import cupyx
        
        # Test if we can actually use CuPy (this will fail if CUDA libs are missing)
        _ = cp.array([1, 2, 3])
        
        _CUPY_AVAILABLE = True
        _CUPY_MODULE = cp
        _CUPYX_MODULE = cupyx
        
    except (ImportError, Exception) as e:
        _CUPY_AVAILABLE = False
        _CUPY_MODULE = None
        _CUPYX_MODULE = None
        warnings.warn(f"CuPy not available: {e}. GPU functionality will be disabled.", 
                     UserWarning, stacklevel=2)
    
    return _CUPY_AVAILABLE


def _check_gpu_available() -> bool:
    """Check if GPU devices are available for computation."""
    global _GPU_AVAILABLE
    
    if _GPU_AVAILABLE is not None:
        return _GPU_AVAILABLE
    
    if not _check_cupy_available():
        _GPU_AVAILABLE = False
        return _GPU_AVAILABLE
    
    try:
        import cupy as cp
        _ = cp.cuda.runtime.getDeviceCount()
        _GPU_AVAILABLE = True
    except Exception:
        _GPU_AVAILABLE = False
    
    return _GPU_AVAILABLE


def get_cupy():
    """Get the CuPy module if available, otherwise None."""
    _check_cupy_available()
    return _CUPY_MODULE


def get_cupyx():
    """Get the CuPy extensions module if available, otherwise None."""
    _check_cupy_available()
    return _CUPYX_MODULE


def require_cupy():
    """Require CuPy to be available, raise error if not."""
    if not _check_cupy_available():
        raise ImportError("CuPy is required for this functionality but is not available. "
                         "Please ensure CUDA libraries are properly installed.")
    return _CUPY_MODULE


def require_gpu():
    """Require GPU to be available, raise error if not."""
    if not _check_gpu_available():
        raise RuntimeError("GPU is required for this functionality but is not available.")


def is_cupy_available() -> bool:
    """Check if CuPy is available."""
    return _check_cupy_available()


def is_gpu_available() -> bool:
    """Check if GPU is available."""
    return _check_gpu_available()


class LazyImport:
    """Lazy import wrapper that delays import until first access."""
    
    def __init__(self, module_name: str, error_message: Optional[str] = None):
        self._module_name = module_name
        self._module = None
        self._error_message = error_message or f"Module {module_name} is not available"
    
    def __getattr__(self, name):
        if self._module is None:
            if not _check_cupy_available():
                raise ImportError(self._error_message)
            
            # Import the specific module
            if self._module_name.startswith('cupyx'):
                import cupyx
                parts = self._module_name.split('.')
                self._module = cupyx
                for part in parts[1:]:  # Skip 'cupyx'
                    self._module = getattr(self._module, part)
            elif self._module_name.startswith('cupy'):
                import cupy
                parts = self._module_name.split('.')
                self._module = cupy
                for part in parts[1:]:  # Skip 'cupy'
                    self._module = getattr(self._module, part)
            else:
                raise ImportError(f"Unknown module pattern: {self._module_name}")
        
        return getattr(self._module, name)


# Provide lazy imports for commonly used modules
cupy = LazyImport('cupy', 'CuPy is not available')
cupyx_scipy_ndimage = LazyImport('cupyx.scipy.ndimage', 'CuPy scipy.ndimage is not available')
cupyx_scipy_signal = LazyImport('cupyx.scipy.signal', 'CuPy scipy.signal is not available')
cupyx_scipy_signal_windows = LazyImport('cupyx.scipy.signal.windows', 'CuPy scipy.signal.windows is not available')
cupyx_scipy_fft = LazyImport('cupyx.scipy.fft', 'CuPy scipy.fft is not available')