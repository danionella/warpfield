import cupyx.scipy.signal.windows
import numpy as np
import cupy as cp
import cupyx
import scipy.ndimage
import cupyx.scipy.ndimage
import cupyx.scipy.signal


def dogfilter(vol, sigma_low=1, sigma_high=4, mode="reflect"):
    """Diffference of Gaussians filter

    Args:
        vol (array_like): data to be filtered
        sigma_low (scalar or sequence of scalar): standard deviations
        sigma_high (scalar or sequence of scalar): standard deviations
        mode (str): The array borders are handled according to the given mode

    Returns:
        (array_like): filtered data

    See also:
        cupyx.scipy.ndimage.gaussian_filter
        skimage.filters.difference_of_gaussians
    """
    in_module = vol.__class__.__module__
    vol = cp.array(vol, "float32", copy=False)
    out = cupyx.scipy.ndimage.gaussian_filter(vol, sigma_low, mode=mode)
    out -= cupyx.scipy.ndimage.gaussian_filter(vol, sigma_high, mode=mode)
    if in_module == "numpy":
        out = out.get()
    return out


def periodic_smooth_decomposition_nd_rfft(img):
    """
    Decompose ND arrays of 2D images into periodic plus smooth components. This can help with edge artifacts in
    Fourier transforms.

    Args:
        img (cupy.ndarray): input image or volume. The last two axes are treated as the image dimensions.

    Returns:
        cupy.ndarray: periodic component
    """
    # compute border-difference
    B = cp.zeros_like(img)
    B[..., 0, :] = img[..., -1, :] - img[..., 0, :]
    B[..., -1, :] = -B[..., 0, :]
    B[..., :, 0] += img[..., :, -1] - img[..., :, 0]
    B[..., :, -1] -= img[..., :, -1] - img[..., :, 0]

    # real FFT of border difference
    B_rfft = cp.fft.rfftn(B, axes=(-2, -1))
    del B

    # build denom for full grid then slice to half-spectrum
    M, N = img.shape[-2:]
    q = cp.arange(M, dtype="float32").reshape(M, 1)
    r = cp.arange(N, dtype="float32").reshape(1, N)
    denom_full = 2 * cp.cos(2 * np.pi * q / M) + 2 * cp.cos(2 * np.pi * r / N) - 4
    # take only first N//2+1 columns
    denom_half = denom_full[:, : (N // 2 + 1)]
    denom_half[0, 0] = 1  # avoid divide by zero

    # compute smooth in freq domain (half-spectrum)
    B_rfft /= denom_half
    B_rfft[..., 0, 0] = 0

    # invert real FFT back to spatial
    # smooth = cp.fft.irfftn(B_rfft, s=(M, N), axes=(-2, -1))
    # periodic = img - smooth
    tmp = cp.fft.irfftn(B_rfft, s=(M, N), axes=(-2, -1))
    tmp *= -1
    tmp += img
    return tmp


def gausswin(shape, sigma):
    """Create Gaussian window of a given shape and sigma

    Args:
        shape (list or tuple): shape along each dimension
        sigma (list or tuple): sigma along each dimension

    Returns:
        (array_like): Gauss window
    """
    grid = np.indices(shape).astype("float32")
    for dim in range(len(grid)):
        grid[dim] -= shape[dim] // 2
        grid[dim] /= sigma[dim]
    out = np.exp(-(grid**2).sum(0) / 2)
    out /= out.sum()
    # out = np.fft.fftshift(out)
    return out


def gausskernel_sheared(sigma, shear=0, truncate=3):
    """Create Gaussian window of a given shape and sigma. The window is sheared along the first two axes.

    Args:
        sigma (float or tuple of float): Standard deviation for Gaussian kernel.
        shear (float): Shear factor in d_axis0 / d_axis1
        truncate (float): Truncate the filter at this many standard deviations.

    Returns:
        window (array_like): n-dimensional window
    """
    # TODO: consider moving to .unshear

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
        window (array_like): n-dimensional window
    """
    out = 1
    for i in range(len(shape)):
        newshape = np.ones(len(shape), dtype="int")
        newshape[i] = shape[i]
        w = window_func(shape[i])
        out = out * np.reshape(w, newshape)
    return out


def accumarray(coords, shape, weights=None, clip=False):
    """Accumulate values into an array using given coordinates and weights

    Args:
        coords (array_like): 3-by-n array of coordinates
        shape (tuple): shape of the output array
        weights (array_like): weights to be accumulated. If None, all weights are set to 1
        clip (bool): if True, clip coordinates to the shape of the output array, else ignore out-of-bounds coordinates. Default is False.

    Returns:
        accum (array_like): accumulated array of the given shape
    """
    assert coords.shape[0] == 3
    coords = np.round(coords.reshape(3, -1)).astype("int")
    if clip:
        for d in len(shape):
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
    """Infill NaNs in an array using Gaussian basis interpolation

    Args:
        arr (array_like): input array
        sigma (float): standard deviation of the Gaussian basis function
        truncate (float): truncate the filter at this many standard deviations. Note: values outside the truncation may still contain NaNs.
        ignore_warning (bool): if True, ignore warnings about invalid values during division
    """
    nans = np.isnan(arr)
    arr_zeros = arr.copy()
    arr_zeros[nans] = 0
    a = scipy.ndimage.gaussian_filter(np.array(arr_zeros, dtype="float64"), sigma=sigma, truncate=truncate)
    b = scipy.ndimage.gaussian_filter(np.array(~nans, dtype="float64"), sigma=sigma, truncate=truncate)
    if ignore_warning:
        with np.errstate(invalid="ignore"):
            out = (a / b).astype(arr.dtype)
    else:
        out = (a / b).astype(arr.dtype)
    return out


def sliding_block(data, block_size=100, block_stride=1):
    """Create a sliding window/block view into the array with the given block shape and stride. The block slides across all dimensions of the array and extracts subsets of the array at all positions.

    Args:
        data (array_like): Array to create the sliding window view from
        block_size (int or tuple of int): Size of window over each axis that takes part in the sliding block
        block_stride (int or tuple of int): Stride of teh window along each axis

    Returns:
        view (ndarray): Sliding block view of the array.

    See Also:
        numpy.lib.stride_tricks.sliding_window_view
        numpy.lib.stride_tricks.as_strided

    """
    block_stride *= np.ones(data.ndim, dtype="int")
    block_size *= np.ones(data.ndim, dtype="int")
    shape = np.r_[1 + (data.shape - block_size) // block_stride, block_size]
    strides = np.r_[block_stride * data.strides, data.strides]
    xp = cp.get_array_module(data)
    out = xp.lib.stride_tricks.as_strided(data, shape, strides)
    return out


def upsampled_dft_rfftn(
    data: cp.ndarray, upsampled_region_size, upsample_factor: int = 1, axis_offsets=None
) -> cp.ndarray:
    """
    Performs an upsampled inverse DFT on a small region around given offsets,
    taking as input the output of cupy.fft.rfftn (real-to-complex FFT).

    This implements the Guizar‑Sicairos local DFT upsampling: no full zero‑padding,
    just a small m×n patch at subpixel resolution.

    Args:
        data: A real-to-complex FFT array of shape (..., M, Nf),
            where Nf = N//2 + 1 corresponds to an original real image width N.
        upsampled_region_size: Size of the output patch (m, n). If an int is
            provided, the same size is used for both dimensions.
        upsample_factor: The integer upsampling factor in each axis.
        axis_offsets: The center of the patch in original-pixel coordinates
            (off_y, off_x). If None, defaults to (0, 0).

    Returns:
        A complex-valued array of shape (..., m, n) containing the
        upsampled inverse DFT patch.
    """
    if data.ndim < 2:
        raise ValueError("Input must have at least 2 dimensions")
    *batch_shape, M, Nf = data.shape
    # determine patch size
    if isinstance(upsampled_region_size, int):
        m, n = upsampled_region_size, upsampled_region_size
    else:
        m, n = upsampled_region_size
    # full width of original image
    N = (Nf - 1) * 2

    # default offset: origin
    off_y, off_x = (0.0, 0.0) if axis_offsets is None else axis_offsets

    # reconstruct full complex FFT via Hermitian symmetry
    full = cp.empty(batch_shape + [M, N], dtype=cp.complex64)
    full[..., :Nf] = data
    if Nf > 1:
        tail = data[..., :, 1:-1]
        full[..., Nf:] = tail[..., ::-1, ::-1].conj()

    # frequency coordinates
    fy = cp.fft.fftfreq(M)[None, :]  # shape (1, M)
    fx = cp.fft.fftfreq(N)[None, :]  # shape (1, N)

    # sample coordinates around offsets
    y_idx = cp.arange(m) - (m // 2)
    x_idx = cp.arange(n) - (n // 2)
    y_coords = off_y[:, None] + y_idx[None, :] / upsample_factor  # (B, m)
    x_coords = off_x[:, None] + x_idx[None, :] / upsample_factor  # (B, n)

    # Build small inverse‐DFT kernels
    ky = cp.exp(2j * cp.pi * y_coords[:, :, None] * fy[None, :, :]).astype("complex64")
    kx = cp.exp(2j * cp.pi * x_coords[:, :, None] * fx[None, :, :]).astype("complex64")

    # First apply along y: (B,m,M) × (B,M,N) -> (B,m,N)
    out1 = cp.einsum("b m M, b M N -> b m N", ky, full)
    # Then along x: (B,m,N) × (B,n,N)ᵀ -> (B,m,n)
    patch = cp.einsum("b m N, b n N -> b m n", out1, kx)

    return patch.real.reshape(*batch_shape, m, n)


def zoom_chop_pad(
    arr, target_shape=None, scale=(1, 1, 1), soft_edge=(0, 0, 0), shift=(0, 0, 0), flip=(False, False, False), cval=0
):
    """Zooms, softens, flips, shifts, and pads/crops a 3D array to match the target shape.

    The conceptual order is as follows: zoom, soften, flip, shift, crop/pad.

    Args:
        arr (np.ndarray or cp.ndarray): The input array to be transformed
        target_shape (tuple of int): The desired target shape to pad/crop to. Defaults to the shape of the input array.
        scale (tuple): Zoom factors for each axis. Default: (1, 1, 1).
        soft_edge (tuple of int): The size of the soft edge (Tukey envelope) to be applied to the input array, in voxels. Default: (0, 0, 0).
        shift (tuple): Shifts for each axis, in voxels. Default: (0, 0, 0).
        flip (tuple of bool): Whether to flip each axis. Default: (False, False, False).
        cval (int, float): The value to use for padding. Default: 0.

    Returns:
        np.ndarray or cp.ndarray: The transformed array. Dtype is float32.
    """

    was_numpy = isinstance(arr, np.ndarray)

    if target_shape is None:
        target_shape = arr.shape

    if any(s > 0 for s in soft_edge):
        arr = cp.array(arr, dtype="float32", copy=True, order="C")
        scaled_edge = np.array(soft_edge) / np.array(scale)
        arr = soften_edges(arr, soft_edge=scaled_edge, copy=False)
    else:
        arr = cp.array(arr, dtype="float32", copy=False, order="C")

    coords = cp.indices(target_shape, dtype=cp.float32)
    for i in range(len(coords)):
        coords[i] -= target_shape[i] / 2
        coords[i] /= scale[i]
        coords[i] += arr.shape[i] / 2
        if flip[i]:
            coords[i] *= -1
            coords[i] += arr.shape[i] - 1
        coords[i] -= shift[i]
    result = cupyx.scipy.ndimage.map_coordinates(arr, coords, order=1, mode="constant", cval=cval)

    if was_numpy:
        result = result.get()
    return result


def soften_edges(arr, soft_edge=(0, 0, 0), copy=True):
    """Apply a soft Tukey edge to the input array.

    Args:
        arr (np.ndarray or cp.ndarray): The input array
        soft_edge (tuple of int): The size of the soft edge (Tukey envelope) to be applied to the input array, in voxels. Default: (0, 0, 0).
        copy (bool): If True, a copy of the array is made. Default: True.

    Returns:
        np.ndarray or cp.ndarray: The transformed array. Dtype is float32.
    """
    was_numpy = isinstance(arr, np.ndarray)
    input_dtype = arr.dtype
    arr = cp.array(arr, dtype="float32", copy=copy)
    if isinstance(soft_edge, int) or isinstance(soft_edge, float):
        soft_edge = (soft_edge,) * arr.ndim
    soft_edge = np.clip(soft_edge, 0, np.array(arr.shape) / 2)

    for i in range(arr.ndim):
        if soft_edge[i] > 0:
            alpha = 2 * soft_edge[i] / arr.shape[i]
            alpha = np.clip(alpha, 0, 1)
            win = cupyx.scipy.signal.windows.tukey(arr.shape[i], alpha)
            arr *= cp.moveaxis(win[:, None, None], 0, i)

    arr = arr.astype(input_dtype, copy=False)
    if was_numpy:
        arr = arr.get()
    return arr


def zoom(arr, zoom_factors, order=1, mode="constant"):
    """Zooms an array by given factors along each axis.

    Args:
        arr (np.ndarray or cp.ndarray): The input array to be zoomed.
        zoom_factors (tuple of float): Zoom factors for each axis. Values greater than 1 result in a larger output array,
            while values less than 1 result in a smaller array. Divide the physical voxel size of the input array by these values to get the physical voxel size of the output array.
        order (int): The order of the spline interpolation. Default is 1 (linear).

    Returns:
        np.ndarray or cp.ndarray: The zoomed array.
    """
    was_numpy = isinstance(arr, np.ndarray)
    arr = cp.array(arr, dtype="float32", copy=False, order="C")
    out = cupyx.scipy.ndimage.zoom(arr, zoom_factors, order=order)
    if was_numpy:
        out = out.get()
    return out


def match_volumes(fixed, fixed_res, moving, moving_res, order=1, soft_edge=(0, 0, 0), cval=0, res_method="fixed"):
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

    Returns:
        fixed_out (ndarray): The fixed volume, rescaled and padded to the target resolution and physical size.
        moving_out (ndarray): The moving volume, rescaled and padded to the target resolution and physical size.
        target_res (tuple): The target resolution used for both volumes.
    """
    fixed_res = np.array(fixed_res)
    moving_res = np.array(moving_res)
    fixed_shape = np.array(fixed.shape)
    moving_shape = np.array(moving.shape)

    # Compute physical sizes
    fixed_phys = fixed_shape * fixed_res
    moving_phys = moving_shape * moving_res

    # Target: match the larger physical size along each axis
    target_phys = np.maximum(fixed_phys, moving_phys)

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

    # Compute the target shape
    target_shape = np.ceil(target_phys / target_res).astype(int)

    # Rescale fixed
    scale_fixed = fixed_res / target_res
    fixed_out = zoom_chop_pad(
        fixed, target_shape=target_shape, scale=scale_fixed, soft_edge=soft_edge, cval=cval, order=order
    )

    # Rescale moving
    scale_moving = moving_res / target_res
    moving_out = zoom_chop_pad(
        moving, target_shape=target_shape, scale=scale_moving, soft_edge=soft_edge, cval=cval, order=order
    )

    return fixed_out, moving_out, tuple(target_res)


def richardson_lucy_generic(img, convolve_psf, correlate_psf=None, num_iter=5, epsilon=1e-3, beta=0.0, initial_guess=None):
    """Richardson-Lucy deconvolution using arbitrary convolution operations, with optional Biggs acceleration.

    Args:
        img (ArrayLike): input image or volume
        convolve_psf (Callable): function to convolve with PSF. Should take an image and return a convolved image. Ensure that the PSF is non-negative and normalized.
        correlate_psf (Callable): function to correlate with PSF. Defaults to convolve_psf (if PSF is symmetric)
        num_iter (int): number of iterations. Default is 5.
        epsilon (float): small constant to prevent divide-by-zero. Default is 1e-3.
        beta (float): acceleration parameter. Default is 0.0 (no acceleration). Typically, beta is in the range [0, 0.5].

    Returns:
        ndarray: deconvolved image
    """
    epsilon = cp.float32(epsilon)
    img = cp.array(img, dtype="float32", copy=False)
    cp.clip(img, 0, None, out=img)
    if num_iter < 1:
        return img
    if correlate_psf is None:
        correlate_psf = convolve_psf

    if initial_guess is not None:
        assert initial_guess.shape == img.shape, "Initial guess must have the same shape as the input image."
        img_decon = cp.array(initial_guess, dtype="float32", copy=False)
        cp.clip(img_decon, 0, None, out=img_decon)
    else:   
        img_decon = img.copy()
    img_decon += epsilon

    for i in range(num_iter):
        img_decon *= correlate_psf(img / (convolve_psf(img_decon) + epsilon))

        if beta > 0:
            if i == 0: 
                img_decon_prev = img_decon.copy()
            else:
                img_decon_new = img_decon.copy()
                img_decon += beta * (img_decon - img_decon_prev)
                cp.clip(img_decon, epsilon, None, out=img_decon)
                img_decon_prev = img_decon_new
    
    return img_decon


def richardson_lucy_fft(img, psf, num_iter=5, epsilon=1e-3, beta=0.0, initial_guess=None):
    """Richardson-Lucy deconvolution using FFT-based convolution and optional Biggs acceleration.

    Args:
        img (ndarray): input image or volume
        psf (ndarray): point spread function (before fftshift)
        num_iter (int): number of iterations
        epsilon (float): small constant to avoid divide-by-zero
        beta (float): Biggs acceleration parameter (0 = no acceleration)

    Returns:
        ndarray: deconvolved image
    """
    psf = cp.array(psf, dtype="float32")
    cp.clip(psf, 0, None, out=psf)
    psf /= psf.sum()

    shape = img.shape
    psf_ft = cp.fft.rfftn(cp.fft.ifftshift(psf), s=shape)
    psf_ft_conj = cp.conj(psf_ft)

    def convolve(x):
        return cp.fft.irfftn(cp.fft.rfftn(x, s=shape) * psf_ft, s=shape)

    def correlate(x):
        return cp.fft.irfftn(cp.fft.rfftn(x, s=shape) * psf_ft_conj, s=shape)

    out = richardson_lucy_generic(
        img, convolve, correlate, num_iter=num_iter, epsilon=epsilon, beta=beta, initial_guess=initial_guess
    )
    return out


def richardson_lucy_gaussian(img, sigmas, num_iter=5, epsilon=1e-3, beta=0.0, initial_guess=None):
    """Richardson-Lucy deconvolution using Gaussian convolution operations

    Args:
        img (ndarray): input image or volume
        sigmas (list or ndarray): list of Gaussian sigmas along each dimension
        num_iter (int): number of iterations
        epsilon (float): small constant to prevent divide-by-zero
        beta (float): acceleration parameter (0 = no acceleration)

    Returns:
        ndarray: deconvolved image
    """
    conv_with_gauss = lambda x: cupyx.scipy.ndimage.gaussian_filter(x, sigmas)
    out = richardson_lucy_generic(
        img, conv_with_gauss, num_iter=num_iter, epsilon=epsilon, beta=beta, initial_guess=initial_guess
    )
    return out


def richardson_lucy_gaussian_shear(img, sigmas, shear, num_iter=5, epsilon=1e-3, beta=0.0, initial_guess=None):
    """Richardson-Lucy deconvolution using a sheared Gaussian psf

    Args:
        img (ndarray): input image or volume
        sigmas (list or ndarray): list of Gaussian sigmas along each dimension
        shear (scalar): shear ratio
        num_iter (int): number of iterations
        epsilon (float): small constant to prevent divide-by-zero
        beta (float): acceleration parameter (0 = no acceleration)

    Returns:
        ndarray: deconvolved image
    """
    if shear == 0:
        return richardson_lucy_gaussian(img, sigmas, num_iter)

    sigmas = np.array(sigmas)
    gw = cp.array(gausskernel_sheared(sigmas, shear=shear, truncate=4), "float32")
    gw01 = gw.sum(2)[:, :, None]
    gw01 /= gw01.sum()
    gw2 = gw.sum(axis=(0, 1))[None, None, :]
    gw2 /= gw2.sum()
    conv_shear = lambda vol: cupyx.scipy.ndimage.convolve(cupyx.scipy.ndimage.convolve(vol, gw01), gw2)
    out = richardson_lucy_generic(
        img, conv_shear, num_iter=num_iter, epsilon=epsilon, beta=beta, initial_guess=initial_guess
    )
    return out
