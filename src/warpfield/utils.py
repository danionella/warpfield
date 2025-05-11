import warnings
from functools import partial
import io

import numpy as np
import cupy as cp
import cupyx.scipy.ndimage

import h5py
import hdf5plugin


def load_data(file_path: str):
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

    elif file_path.endswith(".h5") or file_path.endswith(".hdf5") or file_path.endswith(".mat"):
        raise ValueError(
            f"File path {file_path} was provided without dataset key (example: /path/to/file.h5:dataset_name)"
        )

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

    elif file_path.endswith(".tiff") or file_path.endswith(".tif"):
        try:
            import tifffile
        except ImportError:
            raise ImportError("The 'tifffile' package is required to load TIFF files. Please install it.")
        data = tifffile.imread(file_path)
        tiff_meta = tifffile.TiffFile(file_path).pages[0].tags._dict  # Extract TIFF metadata
        return data, dict(filetype="tiff", path=file_path, meta=tiff_meta)

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

    elif ".mat:" in file_path:
        from scipy.io import loadmat

        file_path, variable_name = file_path.split(":", 1)
        mat_data = loadmat(file_path)
        if variable_name not in mat_data:
            raise ValueError(f"Variable '{variable_name}' not found in MATLAB file '{file_path}'")
        data = mat_data[variable_name]
        return data, dict(filetype="matlab", path=file_path, key=key, meta=mat_data[variable_name].attrs)

    else:
        raise ValueError(f"Unsupported file type: {file_path}")


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
    rgb[..., 0] = reference[None]
    rgb[..., 1] = moving * 0.8

    # clip to shape divisible by 2
    rgb = rgb[:, : (rgb.shape[1] - (rgb.shape[1] % 2)), : (rgb.shape[2] - (rgb.shape[2] % 2))]

    vf = r"drawtext=text='# %{n}':x=w-text_w-10:y=h-text_h-10:fontsize=12:fontcolor=white:borderw=1:bordercolor=black,format=yuv420p"
    try: 
        imageio.mimsave(fn, cp.clip(rgb * 255, 0, 255).astype("uint8"), fps=fps, quality=quality, ffmpeg_params=["-vf", vf])
    except Exception:
        imageio.mimsave(fn, cp.clip(rgb * 255, 0, 255).astype("uint8"), fps=fps, quality=quality)


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

    resized_mips = [
        resize_image(mip, (int(size[0] * scale + 0.5), int(size[1] * scale + 0.5)))
        for mip, size in zip(reordered_mips, sizes)
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


def mosaic_callback(num_slices=9, axis=0, transpose=False, units_per_voxel=[1, 1, 1], width=4096, vmax=1, thick=5):
    """
    Create a mosaic of slices from a 3D dataset along a specified axis, adjusting for voxel aspect ratio.

    Args:
        num_slices (int): The number of slices to include in the mosaic.
        axis (int): The axis along which to extract slices.


    Returns:
        np.ndarray: A 2D mosaic image of the selected slices.
    """
    try:
        from skimage.util import montage
    except ImportError:
        raise ImportError("The 'scikit-image' package is required to create mosaics. Please install it.")

    def wrapped(data):
        data = cp.array(data, copy=False, dtype="float32")
        slice_indices = np.linspace(0, data.shape[axis] - 1, num_slices + 2, dtype=int)[1:-1]

        slices = []
        for i in range(len(slice_indices)):
            slices.append(cp.take(data, slice_indices[i] + cp.arange(-thick // 2, thick // 2), axis=axis).max(axis))
        slices = cp.array(slices)
        aspect_ratio = np.array([units_per_voxel[i] for i in range(3) if i != axis])
        slices = slices.get()
        mosaic = montage(slices)
        zoom_factors = min(width / mosaic.shape[1], 1) * aspect_ratio / aspect_ratio[1]
        mosaic = cupyx.scipy.ndimage.zoom(cp.array(mosaic), zoom_factors, order=1) / vmax
        if transpose:
            mosaic = mosaic.T
        return mosaic.get()

    return wrapped


def showvid(filename, width=600, embed=False, loop=True):
    """
    Display a video in a Jupyter notebook.
    Args:
        filename (str): Path to the video file.
        width (int): Width of the video display.
        embed (bool): Whether to embed the video in the notebook.
        loop (bool): Whether to loop the video.
    """
    try:
        from IPython.display import Video, display
    except ImportError:
        raise ImportError(
            "The 'IPython' package is required to display videos. Please install it."
        )

    html_attributes = "controls loop" if loop else "controls"
    display(Video(filename, embed=embed, width=width, html_attributes=html_attributes))


def show_gif(im0, im1, filename=None, zoom=[1, 1], fps=5):
    """
    Display an animated GIF of im0→im1.

    If filename is None (the default), the GIF is kept in memory.
    If filename is a string, the GIF is written to that file.
    """
    try:
        from IPython.display import Image, display
    except ImportError:
        raise ImportError(
            "The 'IPython' package is required to display images. Please install it."
        )
    try:
        import imageio
    except ImportError:
        raise ImportError(
            "The 'imageio' and 'imageio-ffmpeg' packages are required to create videos. "
            + "Please install them via 'conda install imageio imageio-ffmpeg'"
        )

    frames = [np.clip(im0 * 255, 0, 255).astype("uint8"), np.clip(im1 * 255, 0, 255).astype("uint8")]

    if filename is None:
        # write to in-memory buffer
        buf = io.BytesIO()
        imageio.mimsave(buf, frames, format="GIF", fps=fps, loop=0)
        buf.seek(0)
        display(Image(data=buf.getvalue(), format="gif"))
    else:
        # write to disk
        imageio.mimsave(filename, frames, fps=fps, loop=0)
        display(Image(filename=filename))
