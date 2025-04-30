import numpy as np
import cupy as cp
import cupyx


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
    Decompose ND real images into periodic and smooth components using rfftn/irfftn.
    Last two axes are treated as the image dimensions.
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


def richardson_lucy_blind(img, psf=None, num_iter=5, update_psf=False):
    """Richardson-Lucy deconvolution (regular and blind)

    Args:
        img (ndarray): input image or volume
        psf (ndarray): known psf or initial estimate (before fftshift)
        num_iter (int): number of iterations
        update_psf (bool): True for blind deconvolution

    Returns:
        ndarray: deconvolved image
        ndarray: psf
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
