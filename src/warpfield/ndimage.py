import numpy as np
import cupy as cp
import cupyx
import scipy.ndimage


def dogfilter_gpu(vol, sigma_low=1, sigma_high=4, mode="reflect"):
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
    q = cp.arange(M, dtype='float32').reshape(M, 1)
    r = cp.arange(N, dtype='float32').reshape(1, N)
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


def infill_nans(arr, sigma=0.5, truncate=50):
    """Infill NaNs in an array using Gaussian basis interpolation

    Args:
        arr (array_like): input array
        sigma (float): standard deviation of the Gaussian basis function
        truncate (float): truncate the filter at this many standard deviations
    """
    nans = np.isnan(arr)
    arr_zeros = arr.copy()
    arr_zeros[nans] = 0
    a = scipy.ndimage.gaussian_filter(np.array(arr_zeros, dtype="float64"), sigma=sigma, truncate=truncate)
    b = scipy.ndimage.gaussian_filter(np.array(~nans, dtype="float64"), sigma=sigma, truncate=truncate)
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


def richardson_lucy_blind(img, psf=None, num_iter=5, update_psf=False):
    """Richardson-Lucy deconvolution (regular and blind)

    Args:
        img (ndarray): input image or volume
        psf (ndarray): known psf or initial estimate (before fftshift)
        num_iter (int): number of iterations
        update_psf (bool): True for blind deconvolution

    Returns:
        - ndarray: deconvolved image
        - ndarray: psf
    """

    if psf is None and update_psf:
        psf = cp.ones(img.shape, dtype="float32") / img.size
    psf = cp.array(psf, "float32")
    psf /= psf.sum()
    psf = cp.fft.ifftshift(psf)
    psf_ft = cp.fft.rfftn(psf)
    img = cp.array(img, dtype="float32", copy=False)
    img_decon = img.copy()
    img_decon_ft = cp.fft.rfftn(img_decon)
    ratio = cp.ones_like(img_decon)
    ratio_ft = cp.fft.rfftn(ratio)

    for _ in range(num_iter):
        ratio[:] = img / cp.fft.irfftn(img_decon_ft * psf_ft)
        ratio_ft[:] = cp.fft.rfftn(ratio)
        img_decon *= cp.fft.irfftn(ratio_ft * psf_ft.conj())
        img_decon_ft[:] = cp.fft.rfftn(img_decon)
        if update_psf:
            psf *= cp.fft.irfftn(ratio_ft * img_decon_ft.conj())
            psf /= psf.sum()
            psf_ft[:] = cp.fft.rfftn(psf)

    return img_decon, cp.fft.fftshift(psf)


def richardson_lucy_generic(img, convolve_psf, correlate_psf=None, num_iter=5, epsilon=1 / 100):
    """Richardson-Lucy deconvolution using arbitrary convolution operations

    Args:
        img (ndarray): input image or volume
        convolve_psf (function): function that convolves an image with a psf
        correlate_psf (function): function that correlates an image with a psf. If None, it is assumed that the psf is symmetric and the correlation is computed using the convolution.
        num_iter (int): number of iterations

    Returns:
        ndarray: deconvolved image
    """
    img = cp.clip(cp.array(img, dtype="float32", copy=False), 0, None) + np.float32(epsilon)
    if num_iter < 1:
        return img
    if correlate_psf is None:
        correlate_psf = convolve_psf
    img_decon = img.copy()

    for _ in range(num_iter):
        img_decon *= correlate_psf(img / convolve_psf(img_decon))

    return img_decon


def richardson_lucy_gaussian(img, sigmas, num_iter=5):
    """Richardson-Lucy deconvolution using Gaussian convolution operations

    Args:
        img (ndarray): input image or volume
        sigmas (list or ndarray): list of Gaussian sigmas along each dimension
        num_iter (int): number of iterations

    Returns:
        ndarray: deconvolved image
    """
    import cupyx

    conv_with_gauss = lambda x: cupyx.scipy.ndimage.gaussian_filter(x, sigmas)
    return richardson_lucy_generic(img, conv_with_gauss, conv_with_gauss, num_iter)


def richardson_lucy_gaussian_shear(img, sigmas, shear, num_iter=5):
    """Richardson-Lucy deconvolution using a sheared Gaussian psf

    Args:
        img (ndarray): input image or volume
        sigmas (list or ndarray): list of Gaussian sigmas along each dimension
        shear (scalar): shear ratio
        num_iter (int): number of iterations

    Returns:
        ndarray: deconvolved image
    """
    if shear == 0:
        return richardson_lucy_gaussian(img, sigmas, num_iter)

    import cupyx

    sigmas = np.array(sigmas)
    gw = cp.array(gausskernel_sheared(sigmas, shear=shear, truncate=4), "float32")
    gw01 = gw.sum(2)[:, :, None]
    gw01 /= gw01.sum()
    gw2 = gw.sum(axis=(0, 1))[None, None, :]
    gw2 /= gw2.sum()
    conv_shear = lambda vol: cupyx.scipy.ndimage.convolve(cupyx.scipy.ndimage.convolve(vol, gw01), gw2)
    dec = richardson_lucy_generic(img, conv_shear, num_iter=num_iter)
    return dec
