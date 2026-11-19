import mlx.core as mx

# Kernel body - MLX auto-generates the function signature
# Shapes will be passed via template parameters, scale/offset as buffers
_warp_volume_metal_source = """
    uint3 gid = thread_position_in_grid;
    int i = gid.x;
    int j = gid.y;
    int k = gid.z;

    if (i >= out_dim0 || j >= out_dim1 || k >= out_dim2) {
        return;
    }

    float x = float(i) * disp_inv_scale[0] + disp_offset[0];
    float y = float(j) * disp_inv_scale[1] + disp_offset[1];
    float z = float(k) * disp_inv_scale[2] + disp_offset[2];

    // Sample all displacement channels in one interpolation pass.
    x = metal::clamp(x, -1.0f, float(disp_dim0));
    y = metal::clamp(y, -1.0f, float(disp_dim1));
    z = metal::clamp(z, -1.0f, float(disp_dim2));

    int x0 = int(metal::floor(x));
    int y0 = int(metal::floor(y));
    int z0 = int(metal::floor(z));
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    int z1 = z0 + 1;

    float xd = x - float(x0);
    float yd = y - float(y0);
    float zd = z - float(z0);

    float3 c000, c001, c010, c011, c100, c101, c110, c111;
    bool disp_interior = (x0 >= 0 && x1 < disp_dim0 && y0 >= 0 && y1 < disp_dim1 && z0 >= 0 && z1 < disp_dim2);

    if (disp_interior) {
        int sxy = disp_dim1 * disp_dim2;
        int idx000 = x0 * sxy + y0 * disp_dim2 + z0;
        int idx001 = x0 * sxy + y0 * disp_dim2 + z1;
        int idx010 = x0 * sxy + y1 * disp_dim2 + z0;
        int idx011 = x0 * sxy + y1 * disp_dim2 + z1;
        int idx100 = x1 * sxy + y0 * disp_dim2 + z0;
        int idx101 = x1 * sxy + y0 * disp_dim2 + z1;
        int idx110 = x1 * sxy + y1 * disp_dim2 + z0;
        int idx111 = x1 * sxy + y1 * disp_dim2 + z1;

        c000 = float3(disp_x[idx000], disp_y[idx000], disp_z[idx000]);
        c001 = float3(disp_x[idx001], disp_y[idx001], disp_z[idx001]);
        c010 = float3(disp_x[idx010], disp_y[idx010], disp_z[idx010]);
        c011 = float3(disp_x[idx011], disp_y[idx011], disp_z[idx011]);
        c100 = float3(disp_x[idx100], disp_y[idx100], disp_z[idx100]);
        c101 = float3(disp_x[idx101], disp_y[idx101], disp_z[idx101]);
        c110 = float3(disp_x[idx110], disp_y[idx110], disp_z[idx110]);
        c111 = float3(disp_x[idx111], disp_y[idx111], disp_z[idx111]);
    } else {
        auto fetch_disp3 = [&](int xi, int yi, int zi) -> float3 {
            if (xi >= 0 && xi < disp_dim0 && yi >= 0 && yi < disp_dim1 && zi >= 0 && zi < disp_dim2) {
                int idx = xi * disp_dim1 * disp_dim2 + yi * disp_dim2 + zi;
                return float3(disp_x[idx], disp_y[idx], disp_z[idx]);
            }

            // Linear extrapolation (same rule for all 3 channels).
            int xc = metal::clamp(xi, 0, disp_dim0 - 1);
            int xn = (xi < 0) ? xc + 1 : (xi >= disp_dim0) ? xc - 1 : xc;
            int yc = metal::clamp(yi, 0, disp_dim1 - 1);
            int yn = (yi < 0) ? yc + 1 : (yi >= disp_dim1) ? yc - 1 : yc;
            int zc = metal::clamp(zi, 0, disp_dim2 - 1);
            int zn = (zi < 0) ? zc + 1 : (zi >= disp_dim2) ? zc - 1 : zc;
            // Axes of length 1 (e.g. a translation-only field) have no inner neighbour: stay in bounds,
            // which makes the extrapolation constant along that axis.
            xn = metal::clamp(xn, 0, disp_dim0 - 1);
            yn = metal::clamp(yn, 0, disp_dim1 - 1);
            zn = metal::clamp(zn, 0, disp_dim2 - 1);

            int idx0 = xc * disp_dim1 * disp_dim2 + yc * disp_dim2 + zc;
            int idx1 = xn * disp_dim1 * disp_dim2 + yn * disp_dim2 + zn;

            float3 v0 = float3(disp_x[idx0], disp_y[idx0], disp_z[idx0]);
            float3 v1 = float3(disp_x[idx1], disp_y[idx1], disp_z[idx1]);
            return v0 + (v0 - v1);
        };

        c000 = fetch_disp3(x0, y0, z0);
        c001 = fetch_disp3(x0, y0, z1);
        c010 = fetch_disp3(x0, y1, z0);
        c011 = fetch_disp3(x0, y1, z1);
        c100 = fetch_disp3(x1, y0, z0);
        c101 = fetch_disp3(x1, y0, z1);
        c110 = fetch_disp3(x1, y1, z0);
        c111 = fetch_disp3(x1, y1, z1);
    }

    float3 c00 = c000 * (1 - xd) + c100 * xd;
    float3 c01 = c001 * (1 - xd) + c101 * xd;
    float3 c10 = c010 * (1 - xd) + c110 * xd;
    float3 c11 = c011 * (1 - xd) + c111 * xd;

    float3 c0 = c00 * (1 - yd) + c10 * yd;
    float3 c1 = c01 * (1 - yd) + c11 * yd;
    float3 disp = c0 * (1 - zd) + c1 * zd;

    float d0 = disp.x;
    float d1 = disp.y;
    float d2 = disp.z;

    // Inline volume sampling with zero padding
    float src_x = float(i) + d0;
    float src_y = float(j) + d1;
    float src_z = float(k) + d2;

    src_x = metal::clamp(src_x, -1.0f, float(vol_dim0));
    src_y = metal::clamp(src_y, -1.0f, float(vol_dim1));
    src_z = metal::clamp(src_z, -1.0f, float(vol_dim2));

    int vx0 = int(metal::floor(src_x));
    int vy0 = int(metal::floor(src_y));
    int vz0 = int(metal::floor(src_z));
    int vx1 = vx0 + 1;
    int vy1 = vy0 + 1;
    int vz1 = vz0 + 1;

    float vxd = src_x - float(vx0);
    float vyd = src_y - float(vy0);
    float vzd = src_z - float(vz0);

    float vc00, vc01, vc10, vc11;
    bool vol_interior = (vx0 >= 0 && vx1 < vol_dim0 && vy0 >= 0 && vy1 < vol_dim1 && vz0 >= 0 && vz1 < vol_dim2);

    if (vol_interior) {
        long sxy = (long)vol_dim1 * vol_dim2;  // 64-bit: volumes can exceed 2^31 voxels
        long vidx000 = vx0 * sxy + vy0 * vol_dim2 + vz0;
        long vidx001 = vx0 * sxy + vy0 * vol_dim2 + vz1;
        long vidx010 = vx0 * sxy + vy1 * vol_dim2 + vz0;
        long vidx011 = vx0 * sxy + vy1 * vol_dim2 + vz1;
        long vidx100 = vx1 * sxy + vy0 * vol_dim2 + vz0;
        long vidx101 = vx1 * sxy + vy0 * vol_dim2 + vz1;
        long vidx110 = vx1 * sxy + vy1 * vol_dim2 + vz0;
        long vidx111 = vx1 * sxy + vy1 * vol_dim2 + vz1;

        vc00 = vol[vidx000] * (1 - vxd) + vol[vidx100] * vxd;
        vc01 = vol[vidx001] * (1 - vxd) + vol[vidx101] * vxd;
        vc10 = vol[vidx010] * (1 - vxd) + vol[vidx110] * vxd;
        vc11 = vol[vidx011] * (1 - vxd) + vol[vidx111] * vxd;
    } else {
        auto fetch_vol = [&](int xi, int yi, int zi) -> float {
            if (xi >= 0 && xi < vol_dim0 && yi >= 0 && yi < vol_dim1 && zi >= 0 && zi < vol_dim2) {
                return vol[(long)xi * vol_dim1 * vol_dim2 + (long)yi * vol_dim2 + zi];
            }
            return 0.0f;  // Zero padding
        };

        vc00 = fetch_vol(vx0, vy0, vz0) * (1 - vxd) + fetch_vol(vx1, vy0, vz0) * vxd;
        vc01 = fetch_vol(vx0, vy0, vz1) * (1 - vxd) + fetch_vol(vx1, vy0, vz1) * vxd;
        vc10 = fetch_vol(vx0, vy1, vz0) * (1 - vxd) + fetch_vol(vx1, vy1, vz0) * vxd;
        vc11 = fetch_vol(vx0, vy1, vz1) * (1 - vxd) + fetch_vol(vx1, vy1, vz1) * vxd;
    }

    float vc0 = vc00 * (1 - vyd) + vc10 * vyd;
    float vc1 = vc01 * (1 - vyd) + vc11 * vyd;

    long output_idx = (long)i * out_dim1 * out_dim2 + (long)j * out_dim2 + k;
    output[output_idx] = vc0 * (1 - vzd) + vc1 * vzd;
"""

# Compile the Metal kernel
_warp_volume_metal_kernel = mx.fast.metal_kernel(
    name="warp_volume_kernel",
    input_names=["vol", "disp_x", "disp_y", "disp_z", "disp_inv_scale", "disp_offset"],
    output_names=["output"],
    source=_warp_volume_metal_source,
)


def warp_volume_mlx(vol, disp_field, disp_scale, disp_offset, out_shape=None, tpb=(8, 4, 8)):
    """Warp a 3D volume using a displacement field (Metal kernel).

    For each output voxel, the displacement is interpolated from ``disp_field`` and the volume is sampled at
    the displaced position (see ``ArrayBackend.warp_volume``).

    Args:
        vol (mx.array): 3D float32 volume to be warped.
        disp_field (mx.array): 4D float32 array (3-x-y-z) of displacements along x, y, z.
        disp_scale (mx.array): Scaling factors for the displacement field (3 elements).
        disp_offset (mx.array): Offset values for the displacement field (3 elements).
        out_shape (tuple, optional): Shape of the output. Default: ``vol.shape``.
        tpb (tuple, optional): Threads per threadgroup.

    Returns:
        mx.array: Warped 3D volume (a new array; MLX kernels cannot write into existing arrays).
    """
    out_shape = tuple(vol.shape if out_shape is None else out_shape)
    disp_shape = disp_field.shape[1:]
    result = _warp_volume_metal_kernel(
        inputs=[vol, disp_field[0], disp_field[1], disp_field[2], 1.0 / disp_scale, disp_offset],
        output_shapes=[out_shape],
        output_dtypes=[mx.float32],
        grid=out_shape,
        threadgroup=tuple(tpb),
        template=[
            ("vol_dim0", int(vol.shape[0])),
            ("vol_dim1", int(vol.shape[1])),
            ("vol_dim2", int(vol.shape[2])),
            ("disp_dim0", int(disp_shape[0])),
            ("disp_dim1", int(disp_shape[1])),
            ("disp_dim2", int(disp_shape[2])),
            ("out_dim0", int(out_shape[0])),
            ("out_dim1", int(out_shape[1])),
            ("out_dim2", int(out_shape[2])),
        ],
    )
    return result[0]
