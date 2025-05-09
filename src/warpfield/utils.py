import warnings
from functools import partial

import numpy as np
import cupy as cp
import cupyx.scipy.ndimage
import scipy.ndimage

import h5py
import hdf5plugin


def import_data(file_path: str):
    """
    Import data from various file types, including .npy, .nii, .h5, .mat, DICOM, and TIFF.

    Args:
        file_path (str): Path to the file. For HDF5 and MATLAB files, you can specify the group/key or variable
                         using the format '/path/to/file.h5:/group/key' or '/path/to/file.mat:variable_name'.

    Returns:
        - data (np.ndarray): Loaded data as a NumPy array.
        - metadata (dict): Dictionary containing metadata (e.g., scale, orientation, origin).
    """

    if file_path.endswith(".npy"):
        data = np.load(file_path).copy()
        return data, dict(filetype="npy", path=file_path, meta={})

    elif ".h5:" in file_path or ".hdf5:" in file_path:
        split_index = file_path.rfind(":")  # Find the last colon
        file_path, key = file_path[:split_index], file_path[split_index + 1 :]
        with h5py.File(file_path, "r") as f:
            data = np.array(f[key])
            attrs = dict(f[key].attrs)  # Extract attributes as metadata
        return data, dict(filetype="hdf5", path=file_path, key=key, meta=attrs)

    elif file_path.endswith(".h5") or file_path.endswith(".hdf5"):
        raise ValueError(f"HDF5 file path was provided without dataset key (example: /path/to/file.h5:/dataset)")

    elif ".mat:" in file_path:
        from scipy.io import loadmat

        file_path, variable_name = file_path.split(":", 1)
        mat_data = loadmat(file_path)
        if variable_name not in mat_data:
            raise ValueError(f"Variable '{variable_name}' not found in MATLAB file '{file_path}'")
        data = mat_data[variable_name]
        return data, dict(filetype="matlab", path=file_path, key=key, meta=mat_data[variable_name].attrs)

    elif file_path.endswith(".nii") or file_path.endswith(".nii.gz"):
        try:
            import nibabel as nib
        except ImportError:
            raise ImportError("The 'nibabel' package is required to load NIfTI files. Please install it.")
        warnings.warn(
            "The NIfTI loader ignores scale and offset. Please ensure that fixed and moving volumes are in the same scale and orientation."
        )
        nii = nib.load(file_path)
        data = np.asanyarray(nii.get_fdata(), order="C").astype("float32")
        metadata = {"affine": nii.affine, "header": dict(nii.header)}  # Transformation matrix  # Header information
        return data, dict(filetype="nifti", path=file_path, meta=metadata)

    elif file_path.endswith(".dcm"):
        try:
            import pydicom
        except ImportError:
            raise ImportError("The 'pydicom' package is required to load DICOM files. Please install it.")
        warnings.warn(
            "The DICOM loader ignores scale and offset. Please ensure that fixed and moving volumes are in the same scale and orientation."
        )
        dicom = pydicom.dcmread(file_path)
        data = dicom.pixel_array
        metadata = {
            "spacing": getattr(dicom, "PixelSpacing", None),
            "slice_thickness": getattr(dicom, "SliceThickness", None),
            "orientation": getattr(dicom, "ImageOrientationPatient", None),
            "position": getattr(dicom, "ImagePositionPatient", None),
        }
        return data, dict(filetype="dicom", path=file_path, meta=metadata)

    elif file_path.endswith(".tiff") or file_path.endswith(".tif"):
        try:
            import tifffile
        except ImportError:
            raise ImportError("The 'tifffile' package is required to load TIFF files. Please install it.")
        data = tifffile.imread(file_path)
        tiff_meta = tifffile.TiffFile(file_path).pages[0].tags._dict  # Extract TIFF metadata
        return data, dict(filetype="tiff", path=file_path, meta=tiff_meta)

    else:
        raise ValueError(f"Unsupported file type: {file_path}")


def accumarray(coords, shape, weights=None, clip=False):
    """Accumulate values into an array using given coordinates and weights

    Args:
        coords (array_like): 3-by-n array of coordinates
        shape (tuple): shape of the output array
        weights (array_like): weights to be accumulated. If None, all weights are set to 1
        clip (bool): if True, clip coordinates to the shape of the output array, else ignore out-of-bounds coordinates. Default is False.
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


def create_rgb_video(fn, reference, moving, fps=10, quality=9):
    """
    Create an RGB video from three separate channels (R, G, B).

    Args:
        fn (str): Filename for the output video.
        reference (ndarray): 2D stationary reference image data
        moving (ndarray): 3D moving image data


    Returns:
        ndarray: RGB video.
    """
    try:
        import imageio
    except ImportError:
        raise ImportError(
            "The 'imageio' and 'imageio-ffmpeg' packages are required to create videos. "
            + "Please install them via 'conda install imageio imageio-ffmpeg'"
        )

    rgb = np.zeros((*moving.shape, 3), dtype="float32")
    rgb[..., 0] = moving
    rgb[..., 1] = reference[None]

    # clip to shape divisible by 2
    rgb = rgb[:,:(rgb.shape[1] - (rgb.shape[1] % 2)), :(rgb.shape[2] - (rgb.shape[2] % 2))]

    vf = r"drawtext=text='# %{n}':x=w-text_w-10:y=h-text_h-10:fontsize=12:fontcolor=white:borderw=1:bordercolor=black,format=yuv420p"

    imageio.mimsave(fn, cp.clip(rgb * 255, 0, 255).astype("uint8"), fps=fps, quality=quality, ffmpeg_params=["-vf", vf])


def get_mips(data, units_per_voxel=[1, 1, 1], width=800, axes=[0, 1, 2]):
    """
    Generate 3 maximum intensity projections (MIPs) of a 3D array and tile them in a typical MIP view layout,
    rearranged and transposed according to the specified axes.

    Args:
        data (cp.ndarray): A 3D array representing the volume (CuPy array).
        units_per_voxel (list): The physical size of each voxel in the [z, y, x] directions.
        width (int): The maximum width of the output 2D array.
        axes (list): The desired axis order (e.g., [0, 1, 2] for [z, y, x]).

    Returns:
        cp.ndarray: A 2D array containing the tiled MIPs rearranged and transposed according to the specified axes.
    """

    axes = np.array(axes)
    units_per_voxel = np.array(units_per_voxel)

    # Compute the MIPs along the three original axes (z, y, x)
    mips = [cp.max(data, axis=ax) for ax in range(3)]  # [YX, ZX, ZY]

    # Rearrange and transpose the MIPs based on the new axis order
    reordered_mips = []
    sizes = []
    for i, ax in enumerate(axes):
        mip = mips[ax]
        ax2d = axes[axes != ax]
        if np.argsort(ax2d)[0] != 0:
            mip = mip.T
        sz = np.array(mip.shape) * units_per_voxel[ax2d]
        reordered_mips.append(mip)
        sizes.append(sz)

    # Determine the scaling factor to fit within the specified width
    max_physical_width = sizes[0][1] + sizes[2][0]  # Total physical width (x-axis)
    scale = min(width / max_physical_width, 1.0)

    # Resize each MIP to the correct scale
    def resize_image(image, target_shape):
        scale_factors = [target_shape[0] / image.shape[0], target_shape[1] / image.shape[1]]
        return cupyx.scipy.ndimage.zoom(image, scale_factors, order=1)  # Linear interpolation

    resized_mips = [
        resize_image(mip, (int(size[0] * scale), int(size[1] * scale))) for mip, size in zip(reordered_mips, sizes)
    ]

    # Determine the canvas size
    canvas_height = resized_mips[0].shape[0] + resized_mips[1].shape[0]  # y + z
    canvas_width = resized_mips[0].shape[1] + resized_mips[2].shape[0]  # x + z
    canvas = cp.zeros((canvas_height, canvas_width), dtype=data.dtype)

    # Place the MIPs on the canvas
    canvas[: resized_mips[0].shape[0], : resized_mips[0].shape[1]] = resized_mips[0]  # YX (top-left)
    canvas[: resized_mips[2].shape[1], resized_mips[0].shape[1] :] = resized_mips[2].T  # ZY (top-right)
    canvas[resized_mips[0].shape[0] :, : resized_mips[1].shape[1]] = resized_mips[1]  # ZX (bottom-left)

    return canvas


def get_mips(data, units_per_voxel=[1, 1, 1], width=800, axes=[0, 1, 2]):
    """
    Generate 3 maximum intensity projections (MIPs) of a 3D array and tile them in a typical MIP view layout,
    rearranged and transposed according to the specified axes.

    Args:
        data (cp.ndarray): A 3D array representing the volume (CuPy array).
        units_per_voxel (list): The physical size of each voxel in the [z, y, x] directions.
        width (int): The maximum width of the output 2D array.
        axes (list): The desired axis order (e.g., [0, 1, 2] for [z, y, x]).

    Returns:
        cp.ndarray: A 2D array containing the tiled MIPs rearranged and transposed according to the specified axes.
    """
    axes = np.array(axes)
    units_per_voxel = np.array(units_per_voxel)

    # Compute the MIPs along the three original axes (z, y, x)
    mips = [cp.max(data, axis=ax) for ax in range(3)]  # [YX, ZX, ZY]

    # Rearrange and transpose the MIPs based on the new axis order
    reordered_mips = []
    sizes = []
    for i, ax in enumerate(axes):
        mip = mips[ax]
        ax2d = axes[axes != ax]
        if np.argsort(ax2d)[0] != 0:
            mip = mip.T
        sz = np.array(mip.shape) * units_per_voxel[ax2d]
        reordered_mips.append(mip)
        sizes.append(sz)

    # Determine the scaling factor to fit within the specified width
    scale = min(width / (sizes[0][1] + sizes[2][0]), 1 / np.min(units_per_voxel))

    # Resize each MIP to the correct scale
    def resize_image(image, target_shape):
        scale_factors = [target_shape[0] / image.shape[0], target_shape[1] / image.shape[1]]
        return cupyx.scipy.ndimage.zoom(image, scale_factors, order=1)  # Linear interpolation

    resized_mips = [resize_image(mip, (int(size[0] * scale + 0.5), int(size[1] * scale + 0.5))) for mip, size in zip(reordered_mips, sizes)]

    # Determine the canvas size
    canvas_height = resized_mips[0].shape[0] + resized_mips[1].shape[0]  # y + z
    canvas_width = resized_mips[0].shape[1] + resized_mips[2].shape[0]  # x + z
    canvas = cp.zeros((canvas_height, canvas_width), dtype=data.dtype)

    # Place the MIPs on the canvas
    canvas[: resized_mips[0].shape[0], : resized_mips[0].shape[1]] = resized_mips[0]  # YX (top-left)
    canvas[: resized_mips[2].shape[1], resized_mips[0].shape[1] :] = resized_mips[2].T  # ZY (top-right)
    canvas[resized_mips[0].shape[0] :, : resized_mips[1].shape[1]] = resized_mips[1]  # ZX (bottom-left)

    return canvas


def mips_callback(vmax=1, units_per_voxel=[1, 1, 1], width=800, axes=[0, 1, 2]):
    """
    Return Callback function to generate MIPs for a given volume.

    Args:
        vmax (float): The maximum value for scaling the MIPs.
        units_per_voxel (list): The physical size of each voxel in the [z, y, x] directions.
        width (int): The maximum width of the output 2D array.
        axes (list): The desired axis order (e.g., [0, 1, 2] for [z, y, x]).

    Returns:
        function: A function that takes a 3D cupy volume and returns the MIPs as 2D numpy array.
    """

    def wrapped(vol):
        mips = get_mips(vol, units_per_voxel=units_per_voxel, width=width, axes=axes)
        mips = mips / vmax
        return mips.get()

    return wrapped
