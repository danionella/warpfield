import numpy as np
import cupy as cp


_warp_volume_kernel = cp.RawKernel(
    r"""

__device__ int ravel3d(const int * shape, const int i, const int j, const int k){
    return i * shape[1]*shape[2] + j * shape[2] + k;
}

__device__ float trilinear_nearest(const float * arr, const int * arr_shape, float x, float y, float z){
    x = fminf(fmaxf(x,0.0f), (float)arr_shape[0]-1.001);
    y = fminf(fmaxf(y,0.0f), (float)arr_shape[1]-1.001);
    z = fminf(fmaxf(z,0.0f), (float)arr_shape[2]-1.001);
    float x0f,y0f,z0f, c00, c01, c10, c11, c0, c1, c;
    int x1,y1,z1,x0,y0,z0;
    float xd = modff(x, &x0f);
    float yd = modff(y, &y0f);
    float zd = modff(z, &z0f);
    x0 = (int)x0f;
    y0 = (int)y0f;
    z0 = (int)z0f;
    x1 = x0+1;
    y1 = y0+1;
    z1 = z0+1;
    c00 = arr[ravel3d(arr_shape,x0,y0,z0)] * (1-xd) + arr[ravel3d(arr_shape,x1,y0,z0)] * xd;
    c01 = arr[ravel3d(arr_shape,x0,y0,z1)] * (1-xd) + arr[ravel3d(arr_shape,x1,y0,z1)] * xd;
    c10 = arr[ravel3d(arr_shape,x0,y1,z0)] * (1-xd) + arr[ravel3d(arr_shape,x1,y1,z0)] * xd;
    c11 = arr[ravel3d(arr_shape,x0,y1,z1)] * (1-xd) + arr[ravel3d(arr_shape,x1,y1,z1)] * xd;
    c0 = c00*(1.0f-yd) + c10*yd;
    c1 = c01*(1.0f-yd) + c11*yd;
    c = c0*(1.0f-zd) + c1*zd;
    return c;
}

extern "C" __global__ void warp_volume_kernel(const float * arr, const int * arr_shape, const float * disp_field0, const float * disp_field1, const float * disp_field2, const int * disp_field_shape, const float * disp_scale, const float * disp_offset, float * out, const int * out_shape) {
    float x,y,z,d0,d1,d2;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < out_shape[0]; i += blockDim.x * gridDim.x) {
        for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < out_shape[1]; j += blockDim.y * gridDim.y) {
            for (int k = blockIdx.z * blockDim.z + threadIdx.z; k < out_shape[2]; k += blockDim.z * gridDim.z) {
                x = (float)i/disp_scale[0]+disp_offset[0];
                y = (float)j/disp_scale[1]+disp_offset[1];
                z = (float)k/disp_scale[2]+disp_offset[2];
                d0 = trilinear_nearest(disp_field0, disp_field_shape, x, y, z);
                d1 = trilinear_nearest(disp_field1, disp_field_shape, x, y, z);
                d2 = trilinear_nearest(disp_field2, disp_field_shape, x, y, z);
                int idx = ravel3d(out_shape, i,j,k);
                out[idx] = trilinear_nearest(arr, arr_shape, (float)i+d0, (float)j+d1, (float)k+d2);
            }
        }
    }
}
""",
    "warp_volume_kernel",
)


def warp_volume(vol, disp_field, disp_scale, disp_offset, out=None, tpb=[8, 8, 8]):
    """Warp a 3D volume using a displacement field (calling a CUDA kernel).

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
        out (array_like, optional): Output array to store the warped volume.
            If None, a new array is created.
        tpb (list, optional): Threads per block for CUDA kernel execution.
            Defaults to [8, 8, 8].

    Returns:
        array_like: Warped 3D volume.
    """
    was_numpy = isinstance(vol, np.ndarray)
    vol = cp.array(vol, dtype="float32", copy=False, order="C")
    if out is None:
        out = cp.zeros(vol.shape, dtype=vol.dtype)
    assert out.dtype == cp.dtype("float32")
    bpg = np.ceil(np.array(out.shape) / tpb).astype("int").tolist()  # blocks per grid
    _warp_volume_kernel(
        tuple(bpg),
        tuple(tpb),
        (
            vol,
            cp.r_[vol.shape].astype("int32"),
            disp_field[0].astype("float32"),
            disp_field[1].astype("float32"),
            disp_field[2].astype("float32"),
            cp.r_[disp_field.shape[1:]].astype("int32"),
            disp_scale.astype("float32"),
            disp_offset.astype("float32"),
            out,
            cp.r_[out.shape].astype("int32"),
        ),
    )
    if was_numpy:
        out = cp.asnumpy(out)
    return out
