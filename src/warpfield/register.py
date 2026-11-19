import gc
import os
import pathlib
import warnings
from typing import Any, Callable, List, Union

import h5py
import numpy as np
import scipy.signal
from pydantic import BaseModel
from tqdm.auto import tqdm

from .backends import backend_of, get_array_backend, like_input, resolve_backend
from .ndimage import (
    accumarray,
    dogfilter,
    gausskernel_sheared,
    infill_nans,
    ndwindow,
    periodic_smooth_decomposition_nd_rfft,
    soften_edges,
    upsampled_dft_rfftn,
)
from .utils import create_rgb_video

_ArrayType = Any  # numpy, cupy or mlx array


class _BackendArray:
    """Attribute stored as a float32 array of its owner's backend (``owner.B``), converted on assignment.

    The map owns its arrays (as on main): an assigned array that is already on the backend is copied, so in-place
    updates such as ``wm.warp_field *= 0.5`` never change the caller's arrays or another WarpMap's. (Conversion
    from another backend already makes a new array.) Warp fields are small, so the copies are cheap.
    """

    def __set_name__(self, owner, name):
        self.attr = "_" + name

    def __get__(self, obj, objtype=None):
        return self if obj is None else getattr(obj, self.attr)

    def __set__(self, obj, value):
        B = obj.B
        arr = B.asarray(value, B.float32)
        setattr(obj, self.attr, B.copy(arr) if backend_of(value).name == B.name else arr)


class WarpMap:
    """Represents a 3D displacement field

    Args:
        warp_field (array_like): the displacement field data (3-x-y-z)
        block_size (3-element list or array):
        block_stride (3-element list or array):
        ref_shape (tuple): shape of the reference volume
        mov_shape (tuple): shape of the moving volume
        backend (str or ArrayBackend, optional): "cupy", "mlx" or "numpy". Default: the backend of
            ``warp_field`` if it is a CuPy or MLX array, else the default GPU backend.
    """

    warp_field = _BackendArray()
    block_size = _BackendArray()
    block_stride = _BackendArray()

    def __init__(self, warp_field, block_size, block_stride, ref_shape, mov_shape, backend=None):
        self.B = resolve_backend(warp_field, backend)
        self.warp_field = warp_field  # copied as needed by _BackendArray
        self.block_size = block_size
        self.block_stride = block_stride
        self.ref_shape = tuple(int(s) for s in ref_shape)
        self.mov_shape = tuple(int(s) for s in mov_shape)

    def _new(self, warp_field, block_size, block_stride, ref_shape=None, mov_shape=None):
        """New WarpMap on the same backend."""
        ref_shape = self.ref_shape if ref_shape is None else ref_shape
        mov_shape = self.mov_shape if mov_shape is None else mov_shape
        return WarpMap(warp_field, block_size, block_stride, ref_shape, mov_shape, backend=self.B)

    def _hp(self, x):
        """``x`` as a high-precision backend array (float64 where the device has it, as on CuPy)."""
        x = self.B.asarray(x, self.B.float_hp)
        # Lazy backends (MLX): evaluating here, at the start of resize_to/fit_affine, keeps peak memory ~15% lower
        # (measured; later eval points do not have the same effect). Does nothing on CuPy/NumPy.
        self.B.eval(x)
        return x

    def warp(self, vol, out=None):
        """Apply the warp to a volume. Can be thought of as pulling the moving volume to the fixed volume space.

        Args:
            vol (array_like): the volume to be warped
            out (array_like, optional): float32 output array on the backend, of shape ``ref_shape``

        Returns:
            array_like: warped volume (NumPy if ``vol`` was a NumPy array)
        """
        return like_input(self._warp(vol, out=out), vol)

    def _warp(self, vol, out=None):
        """``warp`` for use inside the pipeline: always returns an array on ``self.B``."""
        B = self.B
        if tuple(vol.shape) != self.mov_shape:
            warnings.warn(f"Volume shape {vol.shape} does not match the expected shape {self.mov_shape}.")
        if out is None:
            out = B.xp.zeros(self.ref_shape, dtype=B.float32)
        offset = -self.block_size / self.block_stride / 2
        return B.warp_volume(B.asarray(vol, B.float32), self.warp_field, self.block_stride, offset, out=out)

    def apply(self, *args, **kwargs):
        """Alias of warp method"""
        return self.warp(*args, **kwargs)

    def fit_affine(self, target=None):
        """Fit affine transformation and return new fitted WarpMap

        Args:
            target (dict): dict with keys "warp_field_shape", "block_size", and "block_stride"

        Returns:
            WarpMap:
            array: affine tranformation coefficients
        """
        B, xp = self.B, self.B.xp
        if target is None:
            warp_field_shape = self.warp_field.shape
            block_size, block_stride = self.block_size, self.block_stride
        else:
            warp_field_shape = tuple(target["warp_field_shape"])
            block_size = B.asarray(target["block_size"], B.float32)
            block_stride = B.asarray(target["block_stride"], B.float32)

        ix = xp.reshape(B.indices(self.warp_field.shape[1:], B.float_hp), (3, -1)).T
        ix = ix * self._hp(self.block_stride) + self._hp(self.block_size) / 2
        a = xp.concatenate([ix, xp.ones((ix.shape[0], 1), dtype=B.float_hp)], axis=1)
        b = ix + xp.reshape(self.warp_field, (3, -1)).T
        coeff = B.lstsq(a, b)
        ix_out = xp.reshape(B.indices(warp_field_shape[1:], B.float_hp), (3, -1)).T
        ix_out = ix_out * self._hp(block_stride) + self._hp(block_size) / 2
        linfit = xp.reshape(((ix_out @ (coeff[:3] - xp.eye(3, dtype=B.float_hp))) + coeff[3]).T, warp_field_shape)
        return self._new(linfit, block_size, block_stride), coeff

    def median_filter(self):
        """Apply median filter to the displacement field

        Returns:
            WarpMap: new WarpMap with median filtered displacement field
        """
        warp_field = self.B.median_filter(self.warp_field, size=[1, 3, 3, 3], mode="nearest")
        return self._new(warp_field, self.block_size, self.block_stride)

    def resize_to(self, target):
        """Resize to target WarpMap, using linear interpolation

        Args:
            target (WarpMap or WarpMapper): target to resize to
                or a dict with keys "warp_field_shape", "block_size", and "block_stride"

        Returns:
            WarpMap: resized WarpMap
        """
        B, xp = self.B, self.B.xp
        if isinstance(target, WarpMap):
            t_sh, t_bsz, t_bst = target.warp_field.shape[1:], target.block_size, target.block_stride
        elif isinstance(target, WarpMapper):
            t_sh, t_bsz, t_bst = target.blocks_shape[:3], target.block_size, target.block_stride
        elif isinstance(target, dict):
            t_sh, t_bsz, t_bst = target["warp_field_shape"][1:], target["block_size"], target["block_stride"]
        else:
            raise ValueError("target must be a WarpMap, WarpMapper, or dict")
        t_sh = tuple(int(s) for s in t_sh)
        t_bsz, t_bst = self._hp(t_bsz), self._hp(t_bst)
        bsz, bst = self._hp(self.block_size), self._hp(self.block_stride)
        ix = xp.reshape(B.indices(t_sh, B.float_hp), (3, -1))
        ix = (ix * t_bst[:, None] + (t_bsz - bsz)[:, None] / 2) / bst[:, None]
        dm_r = xp.stack([xp.reshape(B.map_coordinates(self.warp_field[i], ix), t_sh) for i in range(3)])
        return self._new(dm_r, t_bsz, t_bst)

    def chain(self, target):
        """Chain displacement maps

        Args:
            target (WarpMap): WarpMap to be added to existing map

        Returns:
            WarpMap: new WarpMap with chained displacement field
        """
        return self._new(self.warp_field + self.B.asarray(target.warp_field), target.block_size, target.block_stride)

    def invert(self, **kwargs):
        """alias for invert_fast method"""
        return self.invert_fast(**kwargs)

    def invert_fast(self, sigma=0.5, truncate=20):
        """Invert the displacement field using accumulation and Gaussian basis interpolation.

        Args:
            sigma (float): standard deviation for Gaussian basis interpolation
            truncate (float): truncate parameter for Gaussian basis interpolation

        Returns:
            WarpMap: inverted WarpMap
        """
        warp_field = self.B.to_numpy(self.warp_field)
        block_stride = self.B.to_numpy(self.block_stride)
        target_coords = np.indices(warp_field.shape[1:]) + warp_field / block_stride[:, None, None, None]
        wf_shape = np.ceil(np.array(self.mov_shape) / block_stride + 1).astype("int")
        num_coords = accumarray(target_coords, wf_shape)
        inv_field = np.zeros((3, *wf_shape), dtype=warp_field.dtype)
        for i in range(3):
            inv_field[i] = -accumarray(target_coords, wf_shape, weights=warp_field[i].ravel())
            with np.errstate(invalid="ignore"):
                inv_field[i] /= num_coords
            inv_field[i][num_coords == 0] = np.nan
            inv_field[i] = infill_nans(inv_field[i], sigma=sigma, truncate=truncate)
        return self._new(inv_field, self.block_size, self.block_stride, self.mov_shape, self.ref_shape)

    def push_coordinates(self, coords, negative_shifts=False):
        """Push voxel coordinates from fixed to moving space.

        Args:
            coords (array_like): 3D *voxel* coordinates to be warped (3-by-n array)

        Returns:
            array_like: transformed voxel coordinates (NumPy if ``coords`` was a NumPy array)
        """
        B, xp = self.B, self.B.xp
        assert coords.shape[0] == 3
        c = B.asarray(coords, B.float32)
        c_blocked = c / self.block_stride[:, None] - (self.block_size / (2 * self.block_stride))[:, None]
        shifts = xp.stack([B.map_coordinates(self.warp_field[i], c_blocked) for i in range(3)])
        if negative_shifts:
            shifts = -shifts
        return like_input(c + shifts, coords)

    def pull_coordinates(self, coords):
        """Pull voxel coordinates from moving to fixed space (the reverse of `push_coordinates`).

        Involves inversion, followed by pushing coordinates.

        Args:
            coords (array_like): 3D *voxel* coordinates to be warped (3-by-n array)

        Returns:
            array_like: transformed voxel coordinates (NumPy if ``coords`` was a NumPy array)
        """
        # `invert()` already negates the displacement field, so the shifts must not be negated again here
        return self.invert().push_coordinates(coords)

    def jacobian_det(self, units_per_voxel=[1, 1, 1], edge_order=1):
        """
        Compute det J = det(∇φ) for φ(x)=x+u(x), on the grid of the displacement field.

        Args:
            units_per_voxel (3-element list): physical voxel size
            edge_order : passed to gradient (1 or 2)

        Returns:
            detJ: array of the spatial shape of the displacement field
        """
        B, xp = self.B, self.B.xp
        scaling = B.asarray(units_per_voxel, B.float32) * self.block_stride
        coords = B.indices(self.warp_field.shape[1:], B.float32) * scaling[:, None, None, None]
        phi = coords + self.warp_field
        spacing = [float(s) for s in B.to_numpy(scaling)]
        rows = [xp.stack(B.gradient(phi[i], *spacing, edge_order=edge_order), axis=-1) for i in range(3)]
        return B.det(xp.stack(rows, axis=-2))

    def as_ants_image(self, voxel_size_um=1):
        """Convert to ANTsImage.

        Args:
            voxel_size_um (scalar or array): voxel size (default is 1)

        Returns:
            ants.core.ants_image.ANTsImage:
        """
        try:
            import ants
        except ImportError:
            raise ImportError("ANTs is not installed. Please install it using 'pip install antspyx'")

        to_np = self.B.to_numpy
        ants_image = ants.from_numpy(
            to_np(self.warp_field).transpose(1, 2, 3, 0),
            origin=list((to_np(self.block_size) - 1) / 2 * voxel_size_um),
            spacing=list(to_np(self.block_stride) * voxel_size_um),
            has_components=True,
        )
        return ants_image

    def __repr__(self):
        """String representation of the WarpMap object."""
        return (
            f"WarpMap("
            f"warp_field_shape={self.warp_field.shape}, "
            f"block_size={self.B.to_numpy(self.block_size)}, "
            f"block_stride={self.B.to_numpy(self.block_stride)}, "
            f"transformation: {str(self.mov_shape)} --> {str(self.ref_shape)}, "
            f"backend={self.B.name})"
        )

    def to_h5(self, h5_path, group="warp_map", compression="gzip", overwrite=True):
        """
        Save this WarpMap to an HDF5 file.

        Args:
            h5_path (str or os.PathLike): Path to the HDF5 file.
            group (str): Group path inside the HDF5 file to store the WarpMap (created if missing).
            compression (str or None): Dataset compression (e.g., 'gzip', None).
            overwrite (bool): If True, overwrite existing datasets/attrs inside the group.
        """
        to_np = self.B.to_numpy
        with h5py.File(h5_path, "a") as f:
            if overwrite and (group not in (None, "", "/")) and (group in f):
                del f[group]
            if (not overwrite) and (group in f):
                raise ValueError(f"Group '{group}' already exists in {h5_path}. Set 'overwrite=True' to overwrite it.")
            grp = f.require_group(group) if group not in (None, "", "/") else f
            grp.create_dataset("warp_field", data=to_np(self.warp_field), compression=compression)
            grp.create_dataset("block_size", data=to_np(self.block_size))
            grp.create_dataset("block_stride", data=to_np(self.block_stride))
            grp.create_dataset("ref_shape", data=np.array(self.ref_shape, dtype="int64"))
            grp.create_dataset("mov_shape", data=np.array(self.mov_shape, dtype="int64"))
            grp.attrs["class"] = "WarpMap"

    @classmethod
    def from_h5(cls, h5_path, group="warp_map", backend=None):
        """
        Load a WarpMap from an HDF5 file.

        Args:
            h5_path (str or os.PathLike): Path to the HDF5 file.
            group (str): Group path inside the HDF5 file where the WarpMap is stored.
            backend (str or ArrayBackend, optional): backend for the loaded WarpMap (default: default GPU backend)

        Returns:
            WarpMap: The loaded WarpMap object.
        """
        with h5py.File(h5_path, "r") as f:
            grp = f[group]
            warp_field = grp["warp_field"][:]
            block_size = grp["block_size"][:]
            block_stride = grp["block_stride"][:]
            ref_shape = tuple(grp["ref_shape"][:].tolist())
            mov_shape = tuple(grp["mov_shape"][:].tolist())
        return cls(warp_field, block_size, block_stride, ref_shape, mov_shape, backend=backend)


class WarpMapper:
    """Class that estimates warp field using cross-correlation, based on a piece-wise rigid model.

    Args:
        ref_vol (array_like): The reference volume
        block_size (3-element list or numpy.array): shape of blocks, whose rigid displacement is estimated
        block_stride (3-element list or numpy.array): stride (usually identical to block_size)
        proj_method (callable): Projection method, called as ``proj_method(vol_blocks, axis)``
        backend (str or ArrayBackend, optional): backend to compute on (default: backend of ``ref_vol`` if it
            is a CuPy or MLX array, else the default GPU backend)
    """

    def __init__(
        self,
        ref_vol,
        block_size,
        block_stride=None,
        proj_method=None,
        subpixel=4,
        epsilon=1e-6,
        tukey_alpha=0.5,
        backend=None,
    ):
        if np.any(block_size > np.array(ref_vol.shape)):
            raise ValueError(
                f"Block size (currently: {block_size}) must be smaller than the volume shape ({np.array(ref_vol.shape)})."
            )
        self.B = resolve_backend(ref_vol, backend)
        self.proj_method = proj_method
        self.subpixel = subpixel
        self.epsilon = epsilon
        self.tukey_alpha = tukey_alpha
        self.update_reference(ref_vol, block_size, block_stride)
        self.ref_shape = np.array(ref_vol.shape)

    def update_reference(self, ref_vol, block_size, block_stride=None):
        B, xp = self.B, self.B.xp
        block_size = np.array(block_size)
        block_stride = block_size if block_stride is None else np.array(block_stride)
        ref_blocks = B.sliding_block(B.asarray(ref_vol, B.float32), block_size, block_stride)
        self.blocks_shape = ref_blocks.shape
        ref_blocks_proj = [B.asarray(self.proj_method(ref_blocks, iax), B.float32) for iax in [-3, -2, -1]]
        if self.tukey_alpha is not None and self.tukey_alpha > 0:  # alpha 0 is a rectangular window (no change)
            alpha = float(self.tukey_alpha)
            ref_blocks_proj = [
                p
                * B.asarray(
                    ndwindow([1, 1, 1, *p.shape[-2:]], lambda n: scipy.signal.windows.tukey(n, alpha=alpha)), B.float32
                )
                for p in ref_blocks_proj
            ]
        self.ref_blocks_proj_ft_conj = [xp.conj(B.rfft2(p)) for p in ref_blocks_proj]
        self.block_size = block_size
        self.block_stride = block_stride

    def get_displacement(self, vol, smooth_func=None):
        """Estimate the displacement of vol with the reference volume, via piece-wise rigid cross-correlation with the pre-saved blocks.

        Args:
            vol (array_like): Input volume (on the backend)
            smooth_func (callable): Smoothing function to be applied to the cross-correlation volume,
                called as ``smooth_func(xcorr_proj, block_size)``

        Returns:
            WarpMap
        """
        B, xp = self.B, self.B.xp
        vol_blocks = B.sliding_block(vol, self.block_size, self.block_stride)
        vol_blocks_proj = [B.asarray(self.proj_method(vol_blocks, iax), B.float32) for iax in [-3, -2, -1]]
        del vol_blocks

        disp_field = []
        for i in range(3):
            R = B.rfft2(vol_blocks_proj[i]) * self.ref_blocks_proj_ft_conj[i]
            # s: odd block widths need the full width back (irfft2 alone would return one pixel less)
            xcorr_proj = xp.fft.fftshift(B.irfft2(R, s=vol_blocks_proj[i].shape[-2:]), axes=(-2, -1))
            if smooth_func is not None:
                xcorr_proj = B.asarray(smooth_func(xcorr_proj, self.block_size), B.float32)
            c0, c1 = xcorr_proj.shape[-2] // 2, xcorr_proj.shape[-1] // 2
            xcorr_proj[..., c0, c1] += self.epsilon

            i0, j0 = B.argmax_2d(xcorr_proj)
            i0, j0 = i0 - c0, j0 - c1
            del xcorr_proj
            shifts = upsampled_dft_rfftn(
                xp.reshape(R, (-1, *R.shape[-2:])),
                upsampled_region_size=int(self.subpixel * 2 + 1),
                upsample_factor=self.subpixel,
                axis_offsets=(xp.reshape(i0, (-1,)), xp.reshape(j0, (-1,))),
                width=vol_blocks_proj[i].shape[-1],
            )
            del R
            si, sj = B.argmax_2d(shifts)
            si = xp.reshape(si, i0.shape) - shifts.shape[-2] // 2
            sj = xp.reshape(sj, j0.shape) - shifts.shape[-1] // 2
            del shifts
            disp = (i0 + si / self.subpixel, j0 + sj / self.subpixel)
            B.eval(*disp)  # lazy backends: compute now, so this axis's intermediates can be freed
            disp_field.append(disp)

        disp_field = xp.stack(
            [
                disp_field[1][0] + disp_field[2][0],
                disp_field[0][0] + disp_field[2][1],
                disp_field[0][1] + disp_field[1][1],
            ]
        ).astype(B.float32) / 2
        return WarpMap(disp_field, self.block_size, self.block_stride, self.ref_shape, vol.shape, backend=B)


class RegistrationPyramid:
    """A class for performing multi-resolution registration.

    Args:
        ref_vol (array_like): Reference volume
        recipe (Recipe): Registration recipe.
            IMPORTANT: the block size in the last level cannot be larger than the block_size in any previous level.
        reg_mask (array_like): Mask for registration
        backend (str or ArrayBackend, optional): "cupy", "mlx" or "numpy". Default: backend of ``ref_vol`` if it
            is a CuPy or MLX array, else the default GPU backend.
    """

    def __init__(self, ref_vol, recipe, reg_mask=1, backend=None):
        recipe.model_validate(recipe.model_dump())
        self.B = B = resolve_backend(ref_vol, backend)
        self.recipe = recipe
        self.reg_mask = B.asarray(reg_mask, B.float32)
        self.mappers = []
        ref_vol = B.asarray(ref_vol, B.float32)
        self.ref_shape = ref_vol.shape
        if self.recipe.pre_filter is not None:
            # custom filters may return another array type (e.g. NumPy); the pipeline keeps everything on B
            ref_vol = B.asarray(self.recipe.pre_filter(ref_vol, reg_mask=self.reg_mask), B.float32)
        self.mapper_ix = []
        for i in range(len(recipe.levels)):
            if recipe.levels[i].repeats < 1:
                continue
            block_size = np.array(recipe.levels[i].block_size)
            tmp = np.r_[ref_vol.shape] // -block_size
            block_size[block_size < 0] = tmp[block_size < 0]
            if isinstance(recipe.levels[i].block_stride, (int, float)):
                block_stride = (block_size * recipe.levels[i].block_stride).astype("int")
            else:
                block_stride = np.array(recipe.levels[i].block_stride)
            self.mappers.append(
                WarpMapper(
                    ref_vol,
                    block_size,
                    block_stride=block_stride,
                    proj_method=recipe.levels[i].project,
                    tukey_alpha=recipe.levels[i].tukey_ref,
                    backend=B,
                )
            )
            self.mapper_ix.append(i)
        assert len(self.mappers) > 0, "At least one level of registration is required"

    def register_single(self, vol, callback=None, verbose=False):
        """Register a single volume to the reference volume.

        Args:
            vol (array_like): Volume to be registered
            callback (function): Callback function to be called after each level of registration

        Returns:
            - vol (array_like): Registered volume (NumPy if ``vol`` was a NumPy array)
            - warp_map (WarpMap): Displacement field
            - callback_output (list): List of outputs from the callback function
        """
        B = self.B
        vol_in = vol
        vol = B.asarray(vol, B.float32)
        offsets = (np.array(vol.shape) - np.array(self.ref_shape)) / 2
        warp_map = WarpMap(offsets[:, None, None, None], np.ones(3), np.ones(3), self.ref_shape, vol.shape, backend=B)
        warp_map = warp_map.resize_to(self.mappers[-1])
        callback_output = []
        vol_tmp0 = self.recipe.pre_filter(vol, reg_mask=self.reg_mask) if self.recipe.pre_filter is not None else vol
        vol_tmp0 = B.asarray(vol_tmp0, B.float32)  # custom filters may return another array type (e.g. NumPy)
        vol_tmp = warp_map._warp(vol_tmp0)
        min_block_stride = np.min([mapper.block_stride for mapper in self.mappers], axis=0)
        if callback is not None:
            callback_output.append(callback(vol_tmp))

        if np.any(self.mappers[-1].block_stride > min_block_stride):
            warnings.warn(
                "The block stride (in voxels) in the last level should not be larger than the block stride in any previous level (along any axis)."
            )
        for k, mapper in enumerate(tqdm(self.mappers, desc="Levels", disable=not verbose)):
            level = self.recipe.levels[self.mapper_ix[k]]
            for _ in tqdm(range(level.repeats), leave=False, desc="Repeats", disable=not verbose):
                wm = mapper.get_displacement(vol_tmp, smooth_func=level.smooth)
                wm.warp_field *= level.update_rate
                if level.median_filter:
                    wm = wm.median_filter()
                if level.affine:
                    if (np.array(mapper.blocks_shape[:3]) < 2).sum() > 1:
                        raise ValueError(
                            f"Affine fit needs at least two axes with at least 2 blocks! Volume shape: {self.ref_shape}; block size: {mapper.block_size}"
                        )
                    wm, _ = wm.fit_affine(
                        target=dict(
                            warp_field_shape=(3, *self.mappers[-1].blocks_shape[:3]),
                            block_size=self.mappers[-1].block_size,
                            block_stride=self.mappers[-1].block_stride,
                        )
                    )
                else:
                    wm = wm.resize_to(self.mappers[-1])

                warp_map = warp_map.chain(wm)
                vol_tmp = warp_map._warp(vol_tmp0, out=vol_tmp)
                if callback is not None:
                    callback_output.append(callback(vol_tmp))
                # also evaluates the callback outputs: on lazy backends (MLX) an unevaluated output would keep
                # the vol_tmp of its step alive (outputs that are already evaluated cost nothing)
                B.end_step(vol_tmp, warp_map.warp_field, callback_output)
        vol_tmp = warp_map._warp(vol, out=vol_tmp)
        return like_input(vol_tmp, vol_in), warp_map, callback_output

    def clean_up(self):
        """Free cached device memory (e.g. FFT plans)."""
        self.B.clear_cache()


def register_volumes(
    ref, vol, recipe, reg_mask=1, callback=None, verbose=True, video_path=None, vmax=None, backend="auto"
):
    """Register a volume to a reference volume using a registration pyramid.

    Args:
        ref (numpy, cupy or mlx array): Reference volume
        vol (numpy, cupy or mlx array): Volume to be registered
        recipe (Recipe): Registration recipe
        reg_mask (numpy.array): Mask to be multiplied with the reference volume. Default is 1 (no mask)
        callback (function): Callback function to be called on the volume after each iteration. Default is None.
            Can be used to monitor and optimize registration. Example: `callback = warpfield.utils.mips_callback()`.
            The volume passed to the callback is an array on the backend (e.g. a cupy array); return small,
            NumPy outputs to save GPU memory. Callback outputs for each registration step will be returned as a list.
        verbose (bool): If True, show progress bars. Default is True
        video_path (str): Save a video of the registration process, using callback outputs. The callback has to return 2D frames. Default is None.
        vmax (float): Maximum pixel value (to scale video brightness). If none, set to 99.9 percentile of pixel values.
        backend (str): "auto" (default: the first available of "cupy" and "mlx"), "cupy", "mlx" or "numpy" (CPU, slow).

    Returns:
        - numpy, cupy or mlx array (depending on vol input): Registered volume
        - WarpMap: Displacement field
        - list: List of outputs from the callback function
    """
    recipe.model_validate(recipe.model_dump())
    reg = RegistrationPyramid(ref, recipe, reg_mask=reg_mask, backend=backend)
    B = reg.B
    registered_vol, warp_map, cbout = reg.register_single(vol, callback=callback, verbose=verbose)
    reg.clean_up()
    del reg
    gc.collect()

    if video_path is not None:
        try:
            assert cbout[0].ndim == 2, "Callback output must be a 2D array"
            to_np = lambda x: backend_of(x).to_numpy(x)  # noqa: E731  (callbacks may return GPU arrays)
            # the reference frame goes through the same steps as the reference in RegistrationPyramid
            ref = B.asarray(ref, B.float32)
            if recipe.pre_filter is not None:
                ref = B.asarray(recipe.pre_filter(ref, reg_mask=B.asarray(reg_mask, B.float32)), B.float32)
            ref = to_np(callback(ref))
            vmax = np.percentile(ref, 99.9).item() if vmax is None else vmax
            create_rgb_video(video_path, ref / vmax, np.array([to_np(c) for c in cbout]) / vmax, fps=10)
        except (ValueError, AssertionError) as e:
            warnings.warn(f"Video generation failed with error: {e}")
    return registered_vol, warp_map, cbout


class Projector(BaseModel):
    """A class to apply a 2D projection and filters to a volume block.

    Runs on the backend of its input (see ``RegFilter``).

    Parameters:
        max: if True, apply a max filter to the volume block. Default is True
        normalize: if True, normalize projections by the L2 norm (to get correlations, not covariances). Default is False
        dog: if True, apply a DoG filter to the volume block. Default is True
        low: the lower sigma value for the DoG filter. Default is 0.5
        high: the higher sigma value for the DoG filter. Default is 10.0
        periodic_smooth: if True, apply periodic-smooth decomposition to the projections. Default is False
    """

    max: bool = True
    normalize: Union[bool, float] = False
    dog: bool = True
    low: Union[Union[int, float], List[Union[int, float]]] = 0.5
    high: Union[Union[int, float], List[Union[int, float]]] = 10.0
    periodic_smooth: bool = False

    def __call__(self, vol_blocks, axis):
        """Apply a 2D projection and filters to a volume block
        Args:
            vol_blocks (array_like): Blocked volume to be projected (6D dataset, with the first 3 dimensions being blocks and the last 3 dimensions being voxels)
            axis (int): Axis along which to project
        Returns:
            array_like: Projected volume block (5D dataset, with the first 3 dimensions being blocks and the last 2 dimensions being 2D projections)
        """
        B = backend_of(vol_blocks)
        xp = B.xp
        out = vol_blocks.max(axis) if self.max else vol_blocks.mean(axis)
        if self.periodic_smooth:
            out = periodic_smooth_decomposition_nd_rfft(out, backend=B)
        low = np.delete(np.r_[1, 1, 1] * self.low, axis)
        high = np.delete(np.r_[1, 1, 1] * self.high, axis)
        if self.dog:
            out = dogfilter(out, [0, 0, 0, *low], [0, 0, 0, *high], mode="reflect", backend=B)
        elif not np.all(np.array(self.low) == 0):
            out = B.gaussian_filter(out, [0, 0, 0, *low], mode="reflect", truncate=5.0)
        if self.normalize > 0:
            out /= xp.sqrt(xp.sum(out**2, axis=(-2, -1), keepdims=True)) ** self.normalize + 1e-9
        return out


class Smoother(BaseModel):
    """Smooth blocks with a Gaussian kernel.

    Runs on the backend of its input (see ``RegFilter``).

    Args:
        sigmas (list): [sigma0, sigma1, sigma2]. If None, no smoothing is applied.
        shear (float): shear parameter for gaussian kernel. Default is None.
        long_range_ratio (float): long range ratio for double gaussian kernel. Default is 0.05.
    """

    sigmas: Union[float, List[float]] = [1.0, 1.0, 1.0]
    shear: Union[float, None] = None
    long_range_ratio: Union[float, None] = 0.05

    def __call__(self, xcorr_proj, block_size=None):
        """Apply a Gaussian filter to the cross-correlation data
        Args:
            xcorr_proj (array_like): cross-correlation data (5D array, with the first 3 dimensions being the blocks and the last 2 dimensions being the 2D projection)
            block_size (list): shape of blocks, whose rigid displacement is estimated
        Returns:
            array_like: smoothed cross-correlation volume
        """
        B = backend_of(xcorr_proj)
        truncate = 4.0
        if self.sigmas is None:
            return xcorr_proj
        sigmas = [float(s) for s in np.broadcast_to(self.sigmas, 3)]  # a single number applies to all axes
        if self.shear is not None:
            shear_blocks = self.shear * (block_size[1] / block_size[0])
            gw = gausskernel_sheared(sigmas[:2], shear_blocks, truncate=truncate)
            xcorr_proj = B.convolve_leading(xcorr_proj, gw, mode="constant")
            xcorr_proj = B.gaussian_filter(xcorr_proj, [0, 0, sigmas[2], 0, 0], mode="constant", truncate=truncate)
        else:
            xcorr_proj = B.gaussian_filter(xcorr_proj, [*sigmas, 0, 0], mode="constant", truncate=truncate)
        if self.long_range_ratio is not None:
            xcorr_proj *= 1 - self.long_range_ratio
            long_range_sigmas = [*np.array(sigmas) * 5, 0, 0]
            xcorr_proj += (
                B.gaussian_filter(xcorr_proj, long_range_sigmas, mode="constant", truncate=truncate)
                * self.long_range_ratio
            )
        return xcorr_proj


class RegFilter(BaseModel):
    """A class to apply a filter to the volume before registration.

    Like ``Projector`` and ``Smoother``, this runs on the backend of its input: inside a registration the input
    is already on the GPU, but calling ``recipe.pre_filter(vol)`` by hand on a NumPy array runs on the CPU, which
    is slow for large volumes. Pass ``backend="auto"`` (or move ``vol`` to the GPU first) to filter on the GPU.

    Parameters:
        clip_thresh: threshold for clipping the reference volume. Default is 0
        dog: if True, apply a DoG filter to the volume. Default is True
        low: the lower sigma value for the DoG filter. Default is 0.5
        high: the higher sigma value for the DoG filter. Default is 10.0
        soft_edge: size of a soft (Tukey) edge applied to the volume, in voxels. Default is 0
    """

    clip_thresh: float = 0
    dog: bool = True
    low: float = 0.5
    high: float = 10.0
    soft_edge: Union[Union[int, float], List[Union[int, float]]] = 0.0

    def __call__(self, vol, reg_mask=None, backend=None):
        """Apply the filter to the volume
        Args:
            vol (array_like): 3D volume to be filtered
            reg_mask (array): Mask for registration
            backend (str or ArrayBackend, optional): backend to compute on. Default: the backend of ``vol``
                (so NumPy input runs on the CPU; pass e.g. backend="auto" to use the GPU).
        Returns:
            array_like: Filtered volume (NumPy if ``vol`` was a NumPy array)
        """
        B = backend_of(vol) if backend is None else get_array_backend(backend)
        vol_in = vol
        vol = B.xp.maximum(B.asarray(vol, B.float32) - self.clip_thresh, 0)
        if np.any(np.array(self.soft_edge) > 0):
            vol = soften_edges(vol, soft_edge=self.soft_edge, copy=False, backend=B)
        if reg_mask is not None:
            vol *= B.asarray(reg_mask, B.float32)
        if self.dog:
            vol = dogfilter(vol, self.low, self.high, mode="reflect", backend=B)
        return like_input(vol, vol_in)


class LevelConfig(BaseModel):
    """Configuration for each level of the registration pyramid

    Args:
        block_size (list): shape of blocks, whose rigid displacement is estimated
        block_stride (list): stride (usually identical to block_size)
        repeats (int): number of iterations for this level (deisable level by setting repeats to 0)
        smooth (Smoother or None): Smoother object
        project (Projector, callable or None): Projector object. The callable should take a volume block and an axis as input and return a projected volume block.
        tukey_ref (float): if not None, apply a Tukey window to the reference volume (alpha = tukey_ref). Default is 0.5
        affine (bool): if True, apply affine transformation to the displacement field
        median_filter (bool): if True, apply median filter to the displacement field
        update_rate (float): update rate for the displacement field. Default is 1.0. Can be lowered to dampen oscillations.
    """

    block_size: Union[List[int]]
    block_stride: Union[List[int], float] = 1.0
    project: Union[Projector, Callable[[_ArrayType, int], _ArrayType]] = Projector()
    tukey_ref: Union[float, None] = 0.5
    smooth: Union[Smoother, None] = Smoother()
    affine: bool = False
    median_filter: bool = True
    update_rate: float = 1.0
    repeats: int = 5


class Recipe(BaseModel):
    """Configuration for the registration recipe. Recipe is initialized with a single affine level.

    Args:
        reg_filter (RegFilter, callable or None): Filter to be applied to the reference volume
        levels (list): List of LevelConfig objects
    """

    pre_filter: Union[RegFilter, Callable[[_ArrayType], _ArrayType], None] = RegFilter()
    levels: List[LevelConfig] = [
        LevelConfig(block_size=[-1, -1, -1], repeats=3),  # translation level
        LevelConfig(  # affine level
            block_size=[-2, -2, -2],
            block_stride=0.5,
            repeats=10,
            affine=True,
            median_filter=False,
            smooth=Smoother(sigmas=[0.5, 0.5, 0.5]),
        ),
    ]

    def add_level(self, block_size, **kwargs):
        """Add a level to the registration recipe

        Args:
            block_size (list): shape of blocks, whose rigid displacement is estimated
            **kwargs: additional arguments for LevelConfig
        """
        if isinstance(block_size, (int, float)):
            block_size = [block_size] * 3
        if len(block_size) != 3:
            raise ValueError("block_size must be a list of 3 integers")
        self.levels.append(LevelConfig(block_size=block_size, **kwargs))

    def insert_level(self, index, block_size, **kwargs):
        """Insert a level to the registration recipe

        Args:
            index (int): A number specifying in which position to insert the level
            block_size (list): shape of blocks, whose rigid displacement is estimated
            **kwargs: additional arguments for LevelConfig
        """
        if isinstance(block_size, (int, float)):
            block_size = [block_size] * 3
        if len(block_size) != 3:
            raise ValueError("block_size must be a list of 3 integers")
        self.levels.insert(index, LevelConfig(block_size=block_size, **kwargs))

    @classmethod
    def from_yaml(cls, yaml_path):
        """Load a recipe from a YAML file

        Args:
            yaml_path (str): path to the YAML file

        Returns:
            Recipe: Recipe object
        """
        import yaml

        this_file_dir = pathlib.Path(__file__).resolve().parent
        if not os.path.isfile(yaml_path):
            yaml_path = os.path.join(this_file_dir, "recipes", yaml_path)

        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        return cls.model_validate(data)

    def to_yaml(self, yaml_path):
        """Save the recipe to a YAML file

        Args:
            yaml_path (str): path to the YAML file
        """
        import yaml

        with open(yaml_path, "w") as f:
            yaml.dump(self.model_dump(), f)
        print(f"Recipe saved to {yaml_path}")
