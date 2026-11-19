"""Image processing helpers, written once for all backends.

Functions that take an optional ``backend`` argument compute on: the given backend; else the backend of a
CuPy/MLX input array; else the default GPU backend. NumPy input gives NumPy output.
"""

import numpy as np
import scipy.ndimage

from .backends import backend_of, like_input, resolve_backend


def dogfilter(vol, sigma_low=1, sigma_high=4, mode="reflect", backend=None):
    """Difference of Gaussians filter

    Args:
        vol (array_like): data to be filtered
        sigma_low (scalar or sequence of scalar): standard deviations
        sigma_high (scalar or sequence of scalar): standard deviations
        mode (str): The array borders are handled according to the given mode ("reflect" or "constant")
        backend (str or ArrayBackend, optional): backend to compute on (see module docstring)

    Returns:
        (array_like): filtered data

    See also:
        scipy.ndimage.gaussian_filter
        skimage.filters.difference_of_gaussians
    """
    B = resolve_backend(vol, backend)
    x = B.asarray(vol, B.float32)
    out = B.gaussian_filter(x, sigma_low, mode=mode)
    out -= B.gaussian_filter(x, sigma_high, mode=mode)
    return like_input(out, vol)


def periodic_smooth_decomposition_nd_rfft(img, backend=None):
    """
    Decompose ND arrays of 2D images into periodic plus smooth components. This can help with edge artifacts in
    Fourier transforms.

    Args:
        img (array_like): input image or volume. The last two axes are treated as the image dimensions.
        backend (str or ArrayBackend, optional): backend to compute on (see module docstring)

    Returns:
        array_like: periodic component
    """
    B = resolve_backend(img, backend)
    xp = B.xp
    x = B.asarray(img, B.float32)

    # compute border-difference
    Bd = xp.zeros_like(x)
    Bd[..., 0, :] = x[..., -1, :] - x[..., 0, :]
    Bd[..., -1, :] = -Bd[..., 0, :]
    Bd[..., :, 0] += x[..., :, -1] - x[..., :, 0]
    Bd[..., :, -1] -= x[..., :, -1] - x[..., :, 0]

    # real FFT of border difference
    B_rfft = B.rfft2(Bd)
    del Bd

    # build denom for full grid then slice to half-spectrum
    M, N = x.shape[-2:]
    q = xp.reshape(xp.arange(M, dtype=B.float32), (M, 1))
    r = xp.reshape(xp.arange(N, dtype=B.float32), (1, N))
    denom_full = 2 * xp.cos(2 * np.pi * q / M) + 2 * xp.cos(2 * np.pi * r / N) - 4
    denom_half = denom_full[:, : (N // 2 + 1)]
    denom_half[0, 0] = 1  # avoid divide by zero

    # compute smooth in freq domain (half-spectrum)
    B_rfft /= denom_half
    B_rfft[..., 0, 0] = 0

    # invert real FFT back to spatial: periodic = img - smooth
    tmp = B.irfft2(B_rfft, s=(M, N))
    tmp *= -1
    tmp += x
    return like_input(tmp, img)


def gausswin(shape, sigma):
    """Create Gaussian window of a given shape and sigma

    Args:
        shape (list or tuple): shape along each dimension
        sigma (list or tuple): sigma along each dimension

    Returns:
        (numpy.ndarray): Gauss window
    """
    grid = np.indices(shape).astype("float32")
    for dim in range(len(grid)):
        grid[dim] -= shape[dim] // 2
        grid[dim] /= sigma[dim]
    out = np.exp(-(grid**2).sum(0) / 2)
    out /= out.sum()
    return out


def gausskernel_sheared(sigma, shear=0, truncate=3):
    """Create Gaussian window of a given shape and sigma. The window is sheared along the first two axes.

    Args:
        sigma (float or tuple of float): Standard deviation for Gaussian kernel.
        shear (float): Shear factor in d_axis0 / d_axis1
        truncate (float): Truncate the filter at this many standard deviations.

    Returns:
        window (numpy.ndarray): n-dimensional window
    """
    shape = (np.r_[sigma] * truncate * 2).astype("int")
    shape[0] = np.maximum(shape[0], int(np.ceil(shape[1] * np.abs(shear))))
    shape = (shape // 2) * 2 + 1
    grid = np.indices(shape).astype("float32")
    for dim in range(len(grid)):
        grid[dim] -= shape[dim] // 2
        grid[dim] /= sigma[dim]
    grid[0] = grid[0] + shear * grid[1] * sigma[1] / sigma[0]
    out = np.exp(-(grid**2).sum(0) / 2)
    out /= out.sum()
    return out


def ndwindow(shape, window_func):
    """Create a n-dimensional window function

    Args:
        shape (tuple): shape of the window
        window_func (function): window function to be applied to each dimension

    Returns:
        window (numpy.ndarray): n-dimensional window
    """
    out = 1
    for i in range(len(shape)):
        newshape = np.ones(len(shape), dtype="int")
        newshape[i] = shape[i]
        w = window_func(shape[i])
        out = out * np.reshape(w, newshape)
    return out


def accumarray(coords, shape, weights=None, clip=False):
    """Accumulate values into an array using given coordinates and weights (NumPy, on the CPU)

    Args:
        coords (array_like): 3-by-n array of coordinates
        shape (tuple): shape of the output array
        weights (array_like): weights to be accumulated. If None, all weights are set to 1
        clip (bool): if True, clip coordinates to the shape of the output array, else ignore out-of-bounds coordinates. Default is False.

    Returns:
        accum (numpy.ndarray): accumulated array of the given shape
    """
    assert coords.shape[0] == 3
    coords = np.round(coords.reshape(3, -1)).astype("int")
    if clip:
        for d in range(len(shape)):
            coords[d] = np.minimum(np.maximum(coords[d], 0), shape[d] - 1)
    else:
        valid_ix = np.all((coords >= 0) & (coords < np.array(shape)[:, None]), axis=0)
        coords = coords[:, valid_ix]
        if weights is not None:
            weights = weights.ravel()[valid_ix]
    coords_as_ix = np.ravel_multi_index((*coords,), shape).ravel()
    accum = np.bincount(coords_as_ix, minlength=np.prod(shape), weights=weights)
    accum = accum.reshape(shape)
    return accum


def infill_nans(arr, sigma=0.5, truncate=50, ignore_warning=True):
    """Infill NaNs in an array using Gaussian basis interpolation (NumPy, on the CPU)

    Args:
        arr (numpy.ndarray): input array
        sigma (float): standard deviation of the Gaussian basis function
        truncate (float): truncate the filter at this many standard deviations. Note: values outside the truncation may still contain NaNs.
        ignore_warning (bool): if True, ignore warnings about invalid values during division
    """
    nans = np.isnan(arr)
    arr_zeros = arr.copy()
    arr_zeros[nans] = 0
    a = scipy.ndimage.gaussian_filter(np.array(arr_zeros, dtype="float64"), sigma=sigma, truncate=truncate)
    b = scipy.ndimage.gaussian_filter(np.array(~nans, dtype="float64"), sigma=sigma, truncate=truncate)
    with np.errstate(invalid="ignore" if ignore_warning else "warn"):
        out = (a / b).astype(arr.dtype)
    return out


def sliding_block(data, block_size=100, block_stride=1):
    """Create a sliding window/block view into the array with the given block shape and stride. The block slides across all dimensions of the array and extracts subsets of the array at all positions.

    Args:
        data (array_like): C-contiguous array to create the sliding window view from (NumPy, CuPy or MLX)
        block_size (int or tuple of int): Size of window over each axis that takes part in the sliding block
        block_stride (int or tuple of int): Stride of the window along each axis

    Returns:
        view (array_like): Sliding block view of the array, on the same backend as ``data``.

    See Also:
        numpy.lib.stride_tricks.sliding_window_view
        numpy.lib.stride_tricks.as_strided
    """
    block_size = np.asarray(block_size) * np.ones(data.ndim, dtype="int64")
    block_stride = np.asarray(block_stride) * np.ones(data.ndim, dtype="int64")
    if np.any(block_size <= 0) or np.any(block_stride <= 0):
        raise ValueError("block_size and block_stride must be positive along every axis")
    if np.any(block_size > np.array(data.shape)):
        raise ValueError(f"block_size {block_size} must not exceed the data shape {data.shape}")
    return backend_of(data).sliding_block(data, block_size, block_stride)


def upsampled_dft_rfftn(data, upsampled_region_size, upsample_factor: int = 1, axis_offsets=None):
    """
    Performs an upsampled inverse DFT on a small region around given offsets,
    taking as input the output of rfftn (real-to-complex FFT).

    This implements the Guizar‑Sicairos local DFT upsampling: no full zero‑padding,
    just a small m×n patch at subpixel resolution.

    Args:
        data: A real-to-complex FFT array of shape (B, M, Nf), where Nf = N//2 + 1 corresponds to an
            original real image width N (NumPy, CuPy or MLX).
        upsampled_region_size: Size of the output patch (m, n). If an int is
            provided, the same size is used for both dimensions.
        upsample_factor: The integer upsampling factor in each axis.
        axis_offsets: The centers of the patches in original-pixel coordinates
            (off_y, off_x), each of shape (B,). If None, defaults to (0, 0).

    Returns:
        The real part of the upsampled inverse DFT patches, shape (B, m, n), on the same backend as ``data``.
    """
    B = backend_of(data)
    xp = B.xp
    if data.ndim < 2:
        raise ValueError("Input must have at least 2 dimensions")
    *batch_shape, M, Nf = data.shape
    m, n = (upsampled_region_size,) * 2 if isinstance(upsampled_region_size, int) else upsampled_region_size
    N = (Nf - 1) * 2  # full width of original image

    # reconstruct full complex FFT via Hermitian symmetry
    full = B.asarray(data, B.complex64)
    if Nf > 1:
        full = xp.concatenate([full, xp.conj(full[..., :, 1:-1][..., ::-1, ::-1])], axis=-1)
    full = xp.reshape(full, (-1, M, N))

    # sample coordinates around offsets (high precision where the device has it, as on CuPy)
    n_batch = full.shape[0]
    if axis_offsets is None:
        axis_offsets = (np.zeros(n_batch), np.zeros(n_batch))
    off_y = xp.reshape(B.asarray(axis_offsets[0], B.float_hp), (-1,))
    off_x = xp.reshape(B.asarray(axis_offsets[1], B.float_hp), (-1,))
    fy = xp.fft.fftfreq(M).astype(B.float_hp)
    fx = xp.fft.fftfreq(N).astype(B.float_hp)
    y_coords = off_y[:, None] + (xp.arange(m, dtype=B.float_hp) - m // 2)[None, :] / upsample_factor  # (B, m)
    x_coords = off_x[:, None] + (xp.arange(n, dtype=B.float_hp) - n // 2)[None, :] / upsample_factor  # (B, n)

    # small inverse-DFT kernels
    ky = xp.exp(2j * np.pi * y_coords[:, :, None] * fy[None, None, :]).astype(B.complex64)  # (B, m, M)
    kx = xp.exp(2j * np.pi * x_coords[:, :, None] * fx[None, None, :]).astype(B.complex64)  # (B, n, N)

    # first along y: (B,m,M) x (B,M,N) -> (B,m,N); then along x: (B,m,N) x (B,n,N)^T -> (B,m,n)
    patch = xp.einsum("bmN,bnN->bmn", xp.einsum("bmM,bMN->bmN", ky, full), kx)
    return xp.reshape(xp.real(patch), (*batch_shape, m, n))


def _may_share_memory(B, x, arr):
    """Whether ``x = B.asarray(arr, ...)`` may share memory with ``arr``, so that changing ``x`` in place could
    change ``arr``. ``x is arr`` is not enough: conversion can return a view (np.memmap, CuPy), and NumPy views
    MLX arrays without copying."""
    if isinstance(x, np.ndarray) and isinstance(arr, np.ndarray):
        return np.may_share_memory(x, arr)
    # Same backend: asarray may return arr or a view of it. Different backends: converting allocates new memory,
    # except that NumPy views MLX arrays (shared memory on Apple Silicon).
    source = backend_of(arr).name
    return source == B.name or (B.name == "numpy" and source == "mlx")


def soften_edges(arr, soft_edge=0, copy=True, backend=None):
    """Apply a soft Tukey edge to the input array.

    Args:
        arr (array_like): The input array
        soft_edge (int or tuple of int): The size of the soft edge (Tukey envelope) to be applied to the input array, in voxels,
            per axis or for all axes. Default: 0.
        copy (bool): If True, a copy of the array is made. Default: True.
        backend (str or ArrayBackend, optional): backend to compute on (see module docstring)

    Returns:
        array_like: The transformed array, with the dtype of the input.
    """
    B = resolve_backend(arr, backend)
    x = B.asarray(arr, B.float32)
    if copy and _may_share_memory(B, x, arr):
        x = B.copy(x)
    soft_edge = np.clip(np.broadcast_to(soft_edge, (x.ndim,)), 0, np.array(x.shape) / 2)

    for i in range(x.ndim):
        if soft_edge[i] > 0:
            alpha = np.clip(2 * soft_edge[i] / x.shape[i], 0, 1)
            shape = [1] * x.ndim
            shape[i] = x.shape[i]
            x *= B.xp.reshape(B.tukey(x.shape[i], alpha), shape)

    out = like_input(x, arr)
    return out if out.dtype == arr.dtype else out.astype(arr.dtype)


def zoom_chop_pad(
    arr,
    target_shape=None,
    scale=(1, 1, 1),
    soft_edge=(0, 0, 0),
    shift=(0, 0, 0),
    flip=(False, False, False),
    cval=0,
    order=1,
    backend=None,
):
    """Zooms, softens, flips, shifts, and pads/crops a 3D array to match the target shape.

    The conceptual order is as follows: zoom, soften, flip, shift, crop/pad.

    Args:
        arr (array_like): The input array to be transformed
        target_shape (tuple of int): The desired target shape to pad/crop to. Defaults to the shape of the input array.
        scale (tuple): Zoom factors for each axis. Default: (1, 1, 1).
        soft_edge (tuple of int): The size of the soft edge (Tukey envelope) to be applied to the input array, in voxels. Default: (0, 0, 0).
        shift (tuple): Shifts for each axis, in voxels. Default: (0, 0, 0).
        flip (tuple of bool): Whether to flip each axis. Default: (False, False, False).
        cval (int, float): The value to use for padding. Default: 0.
        order (int): Interpolation order. Default: 1 (linear).
        backend (str or ArrayBackend, optional): backend to compute on (see module docstring)

    Returns:
        array_like: The transformed array. Dtype is float32.
    """
    B = resolve_backend(arr, backend)
    if target_shape is None:
        target_shape = arr.shape
    x = B.asarray(arr, B.float32)
    if any(s > 0 for s in soft_edge):
        x = soften_edges(x, soft_edge=np.array(soft_edge) / np.array(scale), copy=True, backend=B)

    # The source coordinate along axis i only depends on the index along axis i, so compute it as a float32
    # vector (same operations as main) and only expand it to the full grid in the final stack. Python floats
    # keep NumPy/CuPy scalars from promoting the grid to float64.
    target_shape = tuple(int(s) for s in target_shape)
    coords = []
    for i, n in enumerate(target_shape):
        c = np.arange(n, dtype=np.float32)
        c -= float(target_shape[i]) / 2
        c /= float(scale[i])
        c += x.shape[i] / 2
        if flip[i]:
            c *= -1
            c += x.shape[i] - 1
        c -= float(shift[i])
        shape = [1] * len(target_shape)
        shape[i] = n
        coords.append(B.xp.broadcast_to(B.asarray(c.reshape(shape)), target_shape))
    result = B.map_coordinates(x, B.xp.stack(coords), order=order, mode="constant", cval=cval)
    return like_input(result, arr)


def zoom(arr, zoom_factors, order=1, mode="constant", backend=None):
    """Zooms an array by given factors along each axis.

    Args:
        arr (array_like): The input array to be zoomed.
        zoom_factors (tuple of float): Zoom factors for each axis. Values greater than 1 result in a larger output array,
            while values less than 1 result in a smaller array. Divide the physical voxel size of the input array by these values to get the physical voxel size of the output array.
        order (int): The order of the spline interpolation. Default is 1 (linear).
        mode (str): Unused; kept for compatibility.
        backend (str or ArrayBackend, optional): backend to compute on (see module docstring)

    Returns:
        array_like: The zoomed array.
    """
    B = resolve_backend(arr, backend)
    out = B.zoom(B.asarray(arr, B.float32), zoom_factors, order=order)
    return like_input(out, arr)


def match_volumes(
    fixed, fixed_res, moving, moving_res, order=1, soft_edge=(0, 0, 0), cval=0, res_method="fixed", backend=None
):
    """
    Rescale and pad the fixed and moving volumes so both have the same physical size and resolution.

    Args:
        fixed (ndarray): Fixed/reference volume.
        fixed_res (tuple): Pixel size (resolution) for fixed, in physical units.
        moving (ndarray): Moving volume.
        moving_res (tuple): Pixel size (resolution) for moving, in physical units.
        order (int): Interpolation order for zooming.
        soft_edge (tuple): Soft edge parameter for padding.
        cval (float): Constant value for padding.
        res_method (str): How to choose the target resolution.
            "fixed" (default): use fixed_res,
            "min": use the finest (smallest) resolution,
            "max": use the coarsest (largest) resolution,
            "mean": use the mean of fixed_res and moving_res.
        backend (str or ArrayBackend, optional): backend to compute on (see module docstring)

    Returns:
        fixed_out (ndarray): The fixed volume, rescaled and padded to the target resolution and physical size.
        moving_out (ndarray): The moving volume, rescaled and padded to the target resolution and physical size.
        target_res (tuple): The target resolution used for both volumes.
    """
    fixed_res = np.array(fixed_res)
    moving_res = np.array(moving_res)

    # Target: match the larger physical size along each axis
    target_phys = np.maximum(np.array(fixed.shape) * fixed_res, np.array(moving.shape) * moving_res)

    # Determine target resolution
    if res_method == "fixed":
        target_res = fixed_res
    elif res_method == "min":
        target_res = np.minimum(fixed_res, moving_res)
    elif res_method == "max":
        target_res = np.maximum(fixed_res, moving_res)
    elif res_method == "mean":
        target_res = (fixed_res + moving_res) / 2
    else:
        raise ValueError(f"Unknown target_res_type: {res_method}")

    target_shape = np.ceil(target_phys / target_res).astype(int)
    kwargs = dict(target_shape=target_shape, soft_edge=soft_edge, cval=cval, order=order, backend=backend)
    fixed_out = zoom_chop_pad(fixed, scale=fixed_res / target_res, **kwargs)
    moving_out = zoom_chop_pad(moving, scale=moving_res / target_res, **kwargs)
    return fixed_out, moving_out, tuple(target_res)


def richardson_lucy_generic(
    img, convolve_psf, correlate_psf=None, num_iter=5, epsilon=1e-3, beta=0.0, initial_guess=None, backend=None
):
    """Richardson-Lucy deconvolution using arbitrary convolution operations, with optional Biggs acceleration.

    Args:
        img (ArrayLike): input image or volume
        convolve_psf (Callable): function to convolve with PSF. Should take an image and return a convolved image. Ensure that the PSF is non-negative and normalized.
        correlate_psf (Callable): function to correlate with PSF. Defaults to convolve_psf (if PSF is symmetric)
        num_iter (int): number of iterations. Default is 5.
        epsilon (float): small constant to prevent divide-by-zero. Default is 1e-3.
        beta (float): acceleration parameter. Default is 0.0 (no acceleration). Typically, beta is in the range [0, 0.5].
        backend (str or ArrayBackend, optional): backend to compute on (see module docstring)

    Returns:
        array_like: deconvolved image (NumPy if ``img`` was a NumPy array)
    """
    B = resolve_backend(img, backend)
    xp = B.xp
    img_in = img
    img = xp.maximum(B.asarray(img, B.float32), 0)
    if num_iter < 1:
        return like_input(img, img_in)
    if correlate_psf is None:
        correlate_psf = convolve_psf

    if initial_guess is not None:
        assert initial_guess.shape == img.shape, "Initial guess must have the same shape as the input image."
        img_decon = xp.maximum(B.asarray(initial_guess, B.float32), 0)
    else:
        img_decon = B.copy(img)
    img_decon += epsilon

    for i in range(num_iter):
        img_decon *= correlate_psf(img / (convolve_psf(img_decon) + epsilon))

        if beta > 0:
            if i == 0:
                img_decon_prev = B.copy(img_decon)
            else:
                img_decon_new = B.copy(img_decon)
                img_decon = xp.maximum(img_decon + beta * (img_decon - img_decon_prev), epsilon)
                img_decon_prev = img_decon_new

    return like_input(img_decon, img_in)


def richardson_lucy_fft(img, psf, num_iter=5, epsilon=1e-3, beta=0.0, initial_guess=None, backend=None):
    """Richardson-Lucy deconvolution using FFT-based convolution and optional Biggs acceleration.

    Args:
        img (ndarray): input image or volume
        psf (ndarray): point spread function (before fftshift)
        num_iter (int): number of iterations
        epsilon (float): small constant to avoid divide-by-zero
        beta (float): Biggs acceleration parameter (0 = no acceleration)
        backend (str or ArrayBackend, optional): backend to compute on (see module docstring)

    Returns:
        array_like: deconvolved image (NumPy if ``img`` was a NumPy array)
    """
    B = resolve_backend(img, backend)
    xp = B.xp
    psf = np.clip(B.to_numpy(psf).astype("float32"), 0, None)
    psf /= psf.sum()

    shape = tuple(img.shape)
    axes = tuple(range(len(shape)))  # MLX needs explicit axes when `s` is given
    psf_ft = xp.fft.rfftn(B.asarray(np.fft.ifftshift(psf), B.float32), s=shape, axes=axes)
    psf_ft_conj = xp.conj(psf_ft)

    def convolve(x):
        return xp.fft.irfftn(xp.fft.rfftn(x, s=shape, axes=axes) * psf_ft, s=shape, axes=axes)

    def correlate(x):
        return xp.fft.irfftn(xp.fft.rfftn(x, s=shape, axes=axes) * psf_ft_conj, s=shape, axes=axes)

    return richardson_lucy_generic(
        img, convolve, correlate, num_iter=num_iter, epsilon=epsilon, beta=beta, initial_guess=initial_guess, backend=B
    )


def richardson_lucy_gaussian(img, sigmas, num_iter=5, epsilon=1e-3, beta=0.0, initial_guess=None, backend=None):
    """Richardson-Lucy deconvolution using Gaussian convolution operations

    Args:
        img (ndarray): input image or volume
        sigmas (list or ndarray): list of Gaussian sigmas along each dimension
        num_iter (int): number of iterations
        epsilon (float): small constant to prevent divide-by-zero
        beta (float): acceleration parameter (0 = no acceleration)
        backend (str or ArrayBackend, optional): backend to compute on (see module docstring)

    Returns:
        array_like: deconvolved image (NumPy if ``img`` was a NumPy array)
    """
    B = resolve_backend(img, backend)
    return richardson_lucy_generic(
        img,
        lambda x: B.gaussian_filter(x, sigmas),
        num_iter=num_iter,
        epsilon=epsilon,
        beta=beta,
        initial_guess=initial_guess,
        backend=B,
    )


def richardson_lucy_gaussian_shear(
    img, sigmas, shear, num_iter=5, epsilon=1e-3, beta=0.0, initial_guess=None, backend=None
):
    """Richardson-Lucy deconvolution using a sheared Gaussian psf

    Args:
        img (ndarray): input image or volume
        sigmas (list or ndarray): list of Gaussian sigmas along each dimension
        shear (scalar): shear ratio
        num_iter (int): number of iterations
        epsilon (float): small constant to prevent divide-by-zero
        beta (float): acceleration parameter (0 = no acceleration)
        backend (str or ArrayBackend, optional): backend to compute on (see module docstring)

    Returns:
        array_like: deconvolved image (NumPy if ``img`` was a NumPy array)
    """
    B = resolve_backend(img, backend)
    if shear == 0:
        return richardson_lucy_gaussian(img, sigmas, num_iter, backend=B)

    gw = gausskernel_sheared(np.array(sigmas), shear=shear, truncate=4).astype("float32")
    gw01 = gw.sum(2)[:, :, None]
    gw01 /= gw01.sum()
    gw2 = gw.sum(axis=(0, 1))[None, None, :]
    gw2 /= gw2.sum()
    conv_shear = lambda vol: B.convolve_leading(B.convolve_leading(vol, gw01, mode="reflect"), gw2, mode="reflect")
    return richardson_lucy_generic(
        img, conv_shear, num_iter=num_iter, epsilon=epsilon, beta=beta, initial_guess=initial_guess, backend=B
    )
