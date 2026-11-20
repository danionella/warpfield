import numpy as np
import cupy as cp


_warp_volume_kernel = cp.RawKernel(
    r"""

// 64-bit flat indices: volumes can have more than 2^31 voxels
__device__ long long ravel3d(const int * shape, const int i, const int j, const int k){
    return ((long long)i * shape[1] + j) * shape[2] + k;
}

// Extrapolation modes
#define EXTRAP_NEAREST 0
#define EXTRAP_ZERO    1
#define EXTRAP_LINEAR  2

__device__ float trilinear_interp(const float* arr, const int* shape,
                                  float x, float y, float z, int mode)
{
    // Clip to extended range [-1, shape] to limit extrapolation
    x = fminf(fmaxf(x,-1.0f), (float)shape[0]);
    y = fminf(fmaxf(y,-1.0f), (float)shape[1]);
    z = fminf(fmaxf(z,-1.0f), (float)shape[2]);

    int x0 = __float2int_rd(x);
    int y0 = __float2int_rd(y);
    int z0 = __float2int_rd(z);
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    int z1 = z0 + 1;
    float xd = x - (float)x0;  // now in [0, 1] when x in [x0, x0+1]
    float yd = y - (float)y0;
    float zd = z - (float)z0;

    // Flattened 3D index
    auto ravel3d = [](const int* shape, int x, int y, int z) -> long long {
        return ((long long)x * shape[1] + y) * shape[2] + z;
    };

    // Inline voxel sampling with extrapolation
    auto fetch = [&](int xi, int yi, int zi) -> float {
        if (xi >= 0 && xi < shape[0] && yi >= 0 && yi < shape[1] && zi >= 0 && zi < shape[2])
            return arr[ravel3d(shape, xi, yi, zi)];
        if (mode == EXTRAP_ZERO) return 0.0f;
        if (mode == EXTRAP_NEAREST) {
            xi = max(0, min(shape[0]-1, xi));
            yi = max(0, min(shape[1]-1, yi));
            zi = max(0, min(shape[2]-1, zi));
            return arr[ravel3d(shape, xi, yi, zi)];
        }
        if (mode == EXTRAP_LINEAR) {
            int xc = max(0, min(shape[0]-1, xi));
            int xn = (xi < 0) ? xc + 1 : (xi >= shape[0]) ? xc - 1 : xc;
            int yc = max(0, min(shape[1]-1, yi));
            int yn = (yi < 0) ? yc + 1 : (yi >= shape[1]) ? yc - 1 : yc;
            int zc = max(0, min(shape[2]-1, zi));
            int zn = (zi < 0) ? zc + 1 : (zi >= shape[2]) ? zc - 1 : zc;
            // Axes of length 1 (e.g. a translation-only field) have no inner neighbour: stay in bounds,
            // which makes the extrapolation constant along that axis.
            xn = max(0, min(shape[0]-1, xn));
            yn = max(0, min(shape[1]-1, yn));
            zn = max(0, min(shape[2]-1, zn));
            float v0 = arr[ravel3d(shape, xc, yc, zc)];
            float v1 = arr[ravel3d(shape, xn, yn, zn)];
            return v0 + (v0 - v1);  // linear extrapolation
        }
        return 0.0f;
    };

    // Trilinear interpolation
    float c00 = fetch(x0, y0, z0) * (1 - xd) + fetch(x1, y0, z0) * xd;
    float c01 = fetch(x0, y0, z1) * (1 - xd) + fetch(x1, y0, z1) * xd;
    float c10 = fetch(x0, y1, z0) * (1 - xd) + fetch(x1, y1, z0) * xd;
    float c11 = fetch(x0, y1, z1) * (1 - xd) + fetch(x1, y1, z1) * xd;

    float c0 = c00 * (1 - yd) + c10 * yd;
    float c1 = c01 * (1 - yd) + c11 * yd;

    return c0 * (1 - zd) + c1 * zd;
}


extern "C" __global__ void warp_volume_kernel(const float * arr, const int * arr_shape, const float * disp_field0, const float * disp_field1, const float * disp_field2, const int * disp_field_shape, const float * disp_scale, const float * disp_offset, float * out, const int * out_shape) {
    float x,y,z,d0,d1,d2;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < out_shape[0]; i += blockDim.x * gridDim.x) {
        for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < out_shape[1]; j += blockDim.y * gridDim.y) {
            for (int k = blockIdx.z * blockDim.z + threadIdx.z; k < out_shape[2]; k += blockDim.z * gridDim.z) {
                x = (float)i/disp_scale[0]+disp_offset[0];
                y = (float)j/disp_scale[1]+disp_offset[1];
                z = (float)k/disp_scale[2]+disp_offset[2];
                d0 = trilinear_interp(disp_field0, disp_field_shape, x, y, z, EXTRAP_LINEAR);
                d1 = trilinear_interp(disp_field1, disp_field_shape, x, y, z, EXTRAP_LINEAR);
                d2 = trilinear_interp(disp_field2, disp_field_shape, x, y, z, EXTRAP_LINEAR);
                long long idx = ravel3d(out_shape, i,j,k);
                out[idx] = trilinear_interp(arr, arr_shape, (float)i+d0, (float)j+d1, (float)k+d2, EXTRAP_ZERO);
            }
        }
    }
}
""",
    "warp_volume_kernel",
)


_TPB = (8, 8, 8)  # threads per block


def warp_volume_cupy(vol, disp_field, disp_scale, disp_offset, out=None):
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

    All inputs are CuPy arrays (``CupyBackend.warp_volume`` converts them); none are copied unless they are not
    C-contiguous float32. A non-contiguous ``out`` (e.g. a strided view) is filled via a temporary array.

    Returns:
        cupy.ndarray: Warped 3D volume.
    """
    f32 = cp.float32
    vol = cp.ascontiguousarray(vol, dtype=f32)
    if out is None:
        out = cp.zeros(vol.shape, dtype=vol.dtype)
    assert out.dtype == cp.dtype("float32")
    target = out
    if not out.flags.c_contiguous:  # the kernel writes `out` as a flat C-ordered array: use a temporary
        out = cp.zeros(out.shape, dtype=f32)
    bpg = np.ceil(np.array(out.shape) / _TPB).astype("int").tolist()  # blocks per grid
    _warp_volume_kernel(
        tuple(bpg),
        _TPB,
        (
            vol,
            cp.r_[vol.shape].astype("int32"),
            cp.ascontiguousarray(disp_field[0], dtype=f32),
            cp.ascontiguousarray(disp_field[1], dtype=f32),
            cp.ascontiguousarray(disp_field[2], dtype=f32),
            cp.r_[disp_field.shape[1:]].astype("int32"),
            cp.ascontiguousarray(disp_scale, dtype=f32),
            cp.ascontiguousarray(disp_offset, dtype=f32),
            out,
            cp.r_[out.shape].astype("int32"),
        ),
    )
    if target is not out:
        target[...] = out
    return target
