"""MLX implementations used by the MLX backend (see ``mlx_backend.py``).

Inputs are MLX arrays. ``MlxBackend`` decides which cases these cover and falls back to SciPy otherwise.
"""

from functools import lru_cache

import mlx.core as mx
import numpy as np
import scipy.ndimage

# Axes up to this length are filtered with a cached dense filter matrix (one matrix multiply per axis); longer
# axes use a windowed 1D convolution, whose cost does not grow with the square of the length.
_MATRIX_MAX_LEN = 512


def _contiguous_strides(shape):
    """Element strides of a C-contiguous array of the given shape."""
    return [int(np.prod(shape[d + 1 :], dtype=np.int64)) for d in range(len(shape))]


# ----------------------------------------------------------------------------------------------- Gaussian filter
def gaussian_filter(x, sigma, mode="reflect", truncate=4.0):
    """Gaussian filter, like ``scipy.ndimage.gaussian_filter``, for modes "reflect" and "constant" (zero).

    Axes with sigma 0 are not filtered.
    """
    if mode not in ("constant", "reflect"):
        raise ValueError(f"Unsupported padding mode '{mode}'. Use 'constant' or 'reflect'.")
    sigma = [sigma] * x.ndim if np.isscalar(sigma) else list(sigma) + [0.0] * (x.ndim - len(sigma))
    if all(s <= 0 for s in sigma):
        return mx.array(x)  # a copy, as SciPy returns: callers may modify the result in place
    for axis, s in enumerate(sigma):
        if s <= 0:
            continue
        n = x.shape[axis]
        if n <= _MATRIX_MAX_LEN:
            x = _matmul_along(x, _filter_matrix(n, float(s), mode, float(truncate)), axis)
        else:
            x = _convolve_along(x, _gaussian_kernel_1d(s, truncate), axis, mode)
    mx.eval(x)  # without this, lazy evaluation keeps the intermediates of all axes alive (higher peak memory)
    return x


@lru_cache(maxsize=32)
def _filter_matrix(n, sigma, mode, truncate):
    """(n, n) matrix M with M @ v equal to the 1D Gaussian filter of v, edges included (exactly as SciPy)."""
    m = scipy.ndimage.gaussian_filter1d(np.eye(n, dtype=np.float32), sigma, axis=0, mode=mode, truncate=truncate)
    return mx.array(m)


def _matmul_along(x, m, axis):
    """Apply the (n, n) matrix ``m`` along ``axis`` of ``x``."""
    shape = x.shape
    n = shape[axis]
    pre, post = int(np.prod(shape[:axis])), int(np.prod(shape[axis + 1 :]))
    return mx.reshape(m @ mx.reshape(x, (pre, n, post)), shape)


def _gaussian_kernel_1d(sigma, truncate=4.0):
    """1D Gaussian kernel, normalized to sum to 1 (same radius as SciPy)."""
    radius = int(truncate * sigma + 0.5)
    x = mx.arange(-radius, radius + 1, dtype=mx.float32)
    kernel = mx.exp(-0.5 * (x / sigma) ** 2)
    return kernel / mx.sum(kernel)


def _convolve_along(x, kernel, axis, mode):
    """1D convolution of ``x`` with ``kernel`` along ``axis`` (modes "constant" and "reflect")."""
    kernel_size = len(kernel)
    if kernel_size <= 1:
        return x
    pad_list = [(0, 0)] * x.ndim
    pad_list[axis] = (kernel_size // 2, kernel_size // 2)
    # mx.pad(mode="symmetric") gives the same values as _pad_reflect but raises peak memory by ~15%
    padded = mx.pad(x, pad_list) if mode == "constant" else _pad_reflect(x, axis, kernel_size // 2)

    # Move the axis to the front and materialize it: mx.as_strided assumes a contiguous layout.
    padded = mx.array(mx.moveaxis(padded, axis, 0), dtype=mx.float32)
    out_len = padded.shape[0] - kernel_size + 1
    rest_shape = padded.shape[1:]

    # Windows of shape (out_len, kernel_size, *rest_shape), weighted and summed over the window axis
    strides = _contiguous_strides(padded.shape)
    windows = mx.as_strided(padded, shape=[out_len, kernel_size, *rest_shape], strides=[strides[0], *strides])
    kernel_broadcast = mx.reshape(kernel[::-1], [1, kernel_size] + [1] * len(rest_shape))

    # Chunk over out_len so the weighted temporary stays below ~256 MB
    bytes_per_row = kernel_size * int(np.prod(rest_shape)) * 4
    chunk_len = max(1, min(out_len, (256 * 1024 * 1024) // max(1, bytes_per_row)))
    conv_chunks = []
    for start in range(0, out_len, chunk_len):
        conv_chunk = mx.sum(windows[start : start + chunk_len] * kernel_broadcast, axis=1)
        mx.eval(conv_chunk)  # DO NOT REMOVE: without it, lazy evaluation builds the full weighted tensor
        conv_chunks.append(conv_chunk)

    result = conv_chunks[0] if len(conv_chunks) == 1 else mx.concatenate(conv_chunks, axis=0)
    return mx.moveaxis(result, 0, axis)


def _pad_reflect(x, axis, pad):
    """Pad ``x`` by ``pad`` on both sides of ``axis``, mirroring with the edge repeated (SciPy's "reflect")."""
    n = x.shape[axis]

    def reflect_index(pos):
        m = pos % (2 * n)  # the mirrored sequence repeats every 2n samples
        return mx.where(m < n, m, 2 * n - 1 - m)

    left = mx.take(x, reflect_index(mx.arange(-pad, 0, dtype=mx.int32)), axis=axis)
    right = mx.take(x, reflect_index(mx.arange(n, n + pad, dtype=mx.int32)), axis=axis)
    return mx.concatenate([left, x, right], axis=axis)


# ------------------------------------------------------------------------------------------ other operations
def convolve_2d_leading(x, kernel):
    """Convolve ``x`` with a 2D kernel over its first two axes, with zero padding.

    Same result as ``scipy.ndimage.convolve(x, kernel[:, :, None, ...], mode="constant")``.

    Args:
        x (mx.array): Input array with at least 2 axes (e.g. 5D with shape [a, b, c, d, e]).
        kernel (mx.array): 2D kernel.
    """
    assert kernel.ndim == 2, "convolve_2d_leading needs a 2D kernel"
    shape = x.shape
    k0, k1 = kernel.shape
    # Flatten all trailing axes into channels: (a, b, c*d*e), then pad the two convolved axes
    x = mx.reshape(x, (shape[0], shape[1], -1))
    x = mx.pad(x, [(k0 // 2, k0 // 2), (k1 // 2, k1 // 2), (0, 0)])
    channels = x.shape[-1]
    # Grouped 2D convolution, one group per channel; flip=True gives convolution, not correlation
    weight = mx.broadcast_to(mx.reshape(kernel, (1, k0, k1, 1)), (channels, k0, k1, 1))
    out = mx.conv_general(x[None], weight, stride=(1, 1), padding=0, groups=channels, flip=True)
    return mx.reshape(out[0], shape)


def map_coordinates(x, coords):
    """Trilinear interpolation of a 3D array at ``coords`` (shape (3, ...)), clamping to the edges.

    Same result as ``scipy.ndimage.map_coordinates(x, coords, order=1, mode="nearest")``.
    """
    D0, D1, D2 = x.shape
    z = mx.clip(coords[0], 0, D0 - 1)
    y = mx.clip(coords[1], 0, D1 - 1)
    w = mx.clip(coords[2], 0, D2 - 1)

    z0, y0, x0 = (mx.floor(c).astype(mx.int32) for c in (z, y, w))
    z1, y1, x1 = mx.minimum(z0 + 1, D0 - 1), mx.minimum(y0 + 1, D1 - 1), mx.minimum(x0 + 1, D2 - 1)
    zd, yd, xd = z - z0.astype(mx.float32), y - y0.astype(mx.float32), w - x0.astype(mx.float32)

    c00 = x[z0, y0, x0] * (1 - xd) + x[z0, y0, x1] * xd
    c01 = x[z0, y1, x0] * (1 - xd) + x[z0, y1, x1] * xd
    c10 = x[z1, y0, x0] * (1 - xd) + x[z1, y0, x1] * xd
    c11 = x[z1, y1, x0] * (1 - xd) + x[z1, y1, x1] * xd
    c0 = c00 * (1 - yd) + c01 * yd
    c1 = c10 * (1 - yd) + c11 * yd
    return c0 * (1 - zd) + c1 * zd


def median_filter_1333(x):
    """Median over 3x3x3 neighbourhoods of each channel of a (C, Z, Y, X) array, repeating edge values.

    Same result as ``scipy.ndimage.median_filter(x, size=[1, 3, 3, 3], mode="nearest")``.
    """
    C, Z, Y, X = x.shape
    padded = mx.pad(x, [(0, 0), (1, 1), (1, 1), (1, 1)], mode="edge")
    # All 27 shifted copies in one preallocated array, then the 14th smallest value
    neighborhoods = mx.zeros((C, Z, Y, X, 27), dtype=x.dtype)
    idx = 0
    for dz in range(3):
        for dy in range(3):
            for dx in range(3):
                neighborhoods[..., idx] = padded[:, dz : dz + Z, dy : dy + Y, dx : dx + X]
                idx += 1
    return mx.sort(neighborhoods, axis=-1)[..., 13]


def sliding_block(x, block_size, block_stride):
    """Sliding block view of a C-contiguous MLX array: shape ``(*n_blocks, *block_size)``."""
    shape = np.array(x.shape)
    block_size = np.asarray(block_size, dtype=np.int64) * np.ones(x.ndim, dtype=np.int64)
    block_stride = np.asarray(block_stride, dtype=np.int64) * np.ones(x.ndim, dtype=np.int64)
    strides = np.array(_contiguous_strides(x.shape))
    out_shape = np.r_[1 + (shape - block_size) // block_stride, block_size]
    return mx.as_strided(x, shape=out_shape.tolist(), strides=np.r_[block_stride * strides, strides].tolist())
