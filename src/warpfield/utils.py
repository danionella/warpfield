import io
import os
import re
import h5py
import warnings
import hdf5plugin
import scipy.ndimage

import numpy as np

from pathlib import Path

from .backends import available_array_backends as get_available_backends  # noqa: F401 (public name)
from .backends import backend_of
from .backends import set_array_backend as set_backend  # noqa: F401 (public name)


def _split_container_path(file_path, extensions, is_dir=False):
    """Split ``file.ext:name`` or ``file.ext/name`` into ``("file.ext", "name")``, for formats that hold several
    datasets (HDF5, MATLAB, Zarr). ``name`` is None if the path ends at the file. Returns None if no extension in
    ``extensions`` matches.

    If several parts of the path end in a matching extension (e.g. a folder ``exp.h5/`` holding ``x.h5``), the first
    one that exists on disk as a file (as a folder if ``is_dir``, for Zarr stores) is taken. If none does but the
    whole path is a file (e.g. ``exp.h5/x.tif``), it is not this format: None.
    """
    # separators: ":" and "/", and "\\" for Windows paths such as str(Path("data.zarr") / "array")
    pattern = r"\.(?:%s)(?=[:/\\]|$)" % "|".join(re.escape(e.lstrip(".")) for e in extensions)
    candidates = []
    for m in re.finditer(pattern, file_path):
        name = file_path[m.end() + 1 :].replace("\\", "/") if m.end() < len(file_path) else None
        candidates.append((file_path[: m.end()], name or None))
    if not candidates:
        return None
    exists = os.path.isdir if is_dir else os.path.isfile
    found = next((c for c in candidates if exists(c[0])), None)
    if found is None and os.path.isfile(file_path):
        return None
    return found or candidates[0]


def _not_a_group(obj, what):
    """Raise if ``obj`` (an h5py or zarr object) is a group rather than an array, naming what it contains."""
    is_group = isinstance(obj, h5py.Group) or type(obj).__name__ == "Group"  # zarr.Group, without importing zarr
    if is_group:
        names = ", ".join(sorted(obj.keys())) or "nothing"
        raise ValueError(f"{what} is a group, not an array. Give one of its members after it: {names}")
    return obj


def load_data(file_path: str):
    """
    Import data from various file types, including .npy, .nii, .h5, .mat, .zarr, DICOM, and TIFF.

    Args:
        file_path (str): Path to the file. For HDF5, MATLAB and Zarr files, add the dataset, variable or array
                         after the file name, separated by ``:`` or ``/``: ``/path/to/file.h5:group/dataset``,
                         ``/path/to/file.h5/group/dataset``, ``/path/to/file.mat:variable`` or
                         ``/path/to/store.zarr/array``. A Zarr store that is itself an array needs no name.

    Returns:
        - data (np.ndarray): Loaded data as a NumPy array.
        - metadata (dict): Dictionary containing metadata (e.g., scale, orientation, origin). ``path`` is the file
          (or Zarr store) on disk and ``key`` the dataset, variable or array inside it.
    """
    if isinstance(file_path, Path):
        file_path = str(file_path)

    hdf5 = _split_container_path(file_path, [".h5", ".hdf5"])
    matlab = _split_container_path(file_path, [".mat"])
    zarr_path = _split_container_path(file_path, [".zarr"], is_dir=True)

    if file_path.endswith(".npy"):
        data = np.load(file_path).copy()
        return data, dict(filetype="npy", path=file_path, meta={})

    elif hdf5 is not None:
        file_path, key = hdf5
        if key is None:
            raise ValueError(f"Add the dataset to the HDF5 path, e.g. {file_path}:dataset or {file_path}/dataset")
        with h5py.File(file_path, "r") as f:
            dset = _not_a_group(f[key], f"{key} in {file_path}")
            data = np.array(dset)
            attrs = dict(dset.attrs)  # Extract attributes as metadata
        return data, dict(filetype="hdf5", path=file_path, key=key, meta=attrs)

    elif file_path.endswith(".nii") or file_path.endswith(".nii.gz"):
        try:
            import nibabel as nib
        except ImportError:
            raise ImportError("The 'nibabel' package is required to load NIfTI files. Please install it.")
        warnings.warn(
            "The NIfTI loader ignores scale and offset. Please ensure that fixed and moving volumes are at the same scale and orientation."
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
            "The DICOM loader ignores scale and offset. Please ensure that fixed and moving volumes are at the same scale and orientation."
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

    elif matlab is not None:
        from scipy.io import loadmat
        from scipy.io.matlab import matfile_version

        file_path, variable_name = matlab
        if variable_name is None:
            raise ValueError(f"Add the variable to the MATLAB path, e.g. {file_path}:variable or {file_path}/variable")
        not_found = ValueError(f"Variable '{variable_name}' not found in MATLAB file '{file_path}'")
        if matfile_version(file_path)[0] == 2:
            # v7.3 (MATLAB's format for variables over 2 GB) is HDF5. MATLAB stores arrays in column-major order,
            # so HDF5 sees the axes reversed: transpose to get the array as MATLAB (and loadmat) shows it.
            with h5py.File(file_path, "r") as f:
                if variable_name not in f:
                    raise not_found
                data = np.array(_not_a_group(f[variable_name], f"{variable_name} in {file_path}")).T
        else:
            mat_data = loadmat(file_path, variable_names=[variable_name])
            if variable_name not in mat_data:
                raise not_found
            data = mat_data[variable_name]
        return data, dict(filetype="matlab", path=file_path, key=variable_name, meta={})

    elif file_path.endswith(".nrrd") or file_path.endswith(".nhdr"):
        try:
            import nrrd
        except ImportError:
            raise ImportError("The 'pynrrd' package is required to load NRRD files. Please install it.")
        data, header = nrrd.read(file_path)
        return data, dict(filetype="nrrd", path=file_path, meta=header)

    elif zarr_path is not None:
        try:
            import zarr
        except ImportError:
            raise ImportError("The 'zarr' package is required to load Zarr files. Please install it.")
        # The store (x.zarr) is what sits on disk, so report it as the path: the CLI's default output then goes next
        # to the store, not inside it.
        file_path, key = zarr_path
        arr = _not_a_group(zarr.open(file_path, mode="r", path=key or ""), f"{key or 'The store'} in {file_path}")
        data = np.array(arr)
        meta = dict(getattr(arr, "attrs", {}))
        return data, dict(filetype="zarr", path=file_path, key=key, meta=meta)

    else:
        raise ValueError(f"Unsupported file type: {file_path}")


_VIDEO_FILTER = (
    r"drawtext=text='# %{n}':x=w-text_w-10:y=h-text_h-10:fontsize=12:fontcolor=white:borderw=1:bordercolor=black,"
    r"format=yuv420p"
)


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
        import imageio_ffmpeg  # Make sure this is also installed
    except ImportError:
        raise ImportError(
            "The 'imageio' and 'imageio-ffmpeg' packages are required to create videos. "
            + "Please install them via 'conda install imageio imageio-ffmpeg'"
        )

    rgb = np.zeros((*moving.shape, 3), dtype="float32")
    rgb[..., 0] = reference[None]
    rgb[..., 1] = moving
    rgb[..., 2] = moving * 0.5 + reference[None] * 0.5

    # clip to shape divisible by 2
    rgb = rgb[:, : (rgb.shape[1] - (rgb.shape[1] % 2)), : (rgb.shape[2] - (rgb.shape[2] % 2))]

    frames = np.clip(rgb * 255, 0, 255).astype("uint8")

    def written(since):  # whether fn was (re)written: a failed ffmpeg run leaves an existing fn untouched
        return os.path.exists(fn) and os.stat(fn).st_mtime_ns != since and os.path.getsize(fn) > 0

    before = os.stat(fn).st_mtime_ns if os.path.exists(fn) else None
    # The frame counter (ffmpeg's drawtext filter) needs fontconfig. Without it (e.g. on many cluster nodes) imageio
    # neither raises nor writes anything, so write again without the counter.
    try:
        imageio.mimsave(fn, frames, fps=fps, quality=quality, ffmpeg_params=["-vf", _VIDEO_FILTER])
        ok = written(before)
    except Exception:
        ok = False
    if not ok:
        try:
            imageio.mimsave(fn, frames, fps=fps, quality=quality)
        except Exception as e:
            raise ValueError(f"ffmpeg could not write the video to {fn}: {e}") from e
        if not written(before):
            raise ValueError(f"ffmpeg could not write the video to {fn}")


def get_mips(data, units_per_voxel=[1, 1, 1], width=800, axes=[0, 1, 2]):
    """
    Generate 3 maximum intensity projections (MIPs) of a 3D array and tile them in a typical MIP view layout,
    rearranged and transposed according to the specified axes.

    Args:
        data (array-like): A 3D array representing the volume (array-like).
        units_per_voxel (list): The physical size of each voxel in the [z, y, x] directions.
        width (int): The maximum width of the output 2D array.
        axes (list): The desired axis order (e.g., [0, 1, 2] for [z, y, x]).

    Returns:
        np.array: A 2D array containing the tiled MIPs rearranged and transposed according to the specified axes.
    """
    axes = np.array(axes)
    units_per_voxel = np.array(units_per_voxel)

    # Compute the MIPs along the three original axes (z, y, x) on the device, then copy only the 2D images
    B = backend_of(data)
    mips = [B.to_numpy(data.max(axis=ax)) for ax in range(3)]  # [YX, ZX, ZY]

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
        return scipy.ndimage.zoom(image, scale_factors, order=1)  # Linear interpolation

    resized_mips = [
        resize_image(mip, (int(size[0] * scale + 0.5), int(size[1] * scale + 0.5)))
        for mip, size in zip(reordered_mips, sizes)
    ]

    # Determine the canvas size
    canvas_height = resized_mips[0].shape[0] + resized_mips[1].shape[0]  # y + z
    canvas_width = resized_mips[0].shape[1] + resized_mips[2].shape[0]  # x + z
    canvas = np.zeros((canvas_height, canvas_width), dtype=mips[0].dtype)

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
        function: A function that takes a 3D volume and returns the MIPs as 2D numpy array.
    """

    def wrapped(vol):
        mips = get_mips(vol, units_per_voxel=units_per_voxel, width=width, axes=axes)
        mips = mips / vmax
        return mips

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
        B = backend_of(data)
        slice_indices = np.linspace(0, data.shape[axis] - 1, num_slices + 2, dtype=int)[1:-1]

        slices = []
        for i in range(len(slice_indices)):
            ix = (slice_indices[i] + np.arange(-thick // 2, thick // 2)) % data.shape[axis]  # wrap, as cupy.take does
            ix = B.asarray(ix, B.int32)
            slices.append(B.to_numpy(B.xp.take(data, ix, axis=axis).max(axis)).astype("float32"))
        slices = np.array(slices)
        aspect_ratio = np.array([units_per_voxel[i] for i in range(3) if i != axis])
        mosaic = montage(slices)
        zoom_factors = min(width / mosaic.shape[1], 1) * aspect_ratio / aspect_ratio[1]
        mosaic = scipy.ndimage.zoom(np.array(mosaic), zoom_factors, order=1) / vmax
        if transpose:
            mosaic = mosaic.T
        return mosaic

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
        raise ImportError("The 'IPython' package is required to display videos. Please install it.")

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
        raise ImportError("The 'IPython' package is required to display images. Please install it.")
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
