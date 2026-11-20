from .backends import current_backend, like_input


def warp_volume(vol, disp_field, disp_scale, disp_offset, out=None):
    """Warp a 3D volume using a displacement field.

    This function applies a displacement field, typically obtained from a
    registration algorithm, to warp a 3D volume. The displacement field
    is a 4D array of shape (3, x, y, z), where the first dimension corresponds
    to the x, y, and z displacements. It defines, for each voxel in the target
    volume, the source location in the warped volume.

    Args:
        vol (array_like): 3D input array (x-y-z) to be warped.
        disp_field (array_like): 4D array (3-x-y-z) of displacements along x, y, z.
        disp_scale (array_like): Scaling factors for the displacement field.
        disp_offset (array_like): Offset values for the displacement field.
        out (array_like, optional): float32 output array on the backend, which also sets the output shape.
            If None, a new array of the shape of ``vol`` is created.

    Returns:
        array_like: Warped 3D volume (NumPy if ``vol`` was a NumPy array).
    """
    B = current_backend()
    res = B.warp_volume(B.asarray(vol, B.float32), disp_field, disp_scale, disp_offset, out=out)
    return like_input(res, vol)
