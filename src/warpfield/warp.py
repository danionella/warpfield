import numpy as np
import cupy as cp

_warp_volume_kernel_code = r"""
__device__ int ravel3d(const int * shape, const int i, const int j, const int k){
    return i*shape[1]*shape[2] + j*shape[2] + k;
}
#define EXTRAP_NEAREST 0
#define EXTRAP_ZERO    1
#define EXTRAP_LINEAR  2
__device__ float trilinear_interp(const float* arr, const int* shape,
                                  float x, float y, float z, int mode)
{
    x = fminf(fmaxf(x,-1.0f),(float)shape[0]);
    y = fminf(fmaxf(y,-1.0f),(float)shape[1]);
    z = fminf(fmaxf(z,-1.0f),(float)shape[2]);
    int x0=__float2int_rd(x), y0=__float2int_rd(y), z0=__float2int_rd(z);
    int x1=x0+1, y1=y0+1, z1=z0+1;
    float xd=x-(float)x0, yd=y-(float)y0, zd=z-(float)z0;
    auto r3 = [&](int xi,int yi,int zi){return (xi*shape[1]+yi)*shape[2]+zi;};
    auto fetch = [&](int xi,int yi,int zi){
        if(0<=xi&&xi<shape[0]&&0<=yi&&yi<shape[1]&&0<=zi&&zi<shape[2])
            return arr[r3(xi,yi,zi)];
        if(mode==EXTRAP_ZERO) return 0.0f;
        if(mode==EXTRAP_NEAREST){
            xi=max(0,min(shape[0]-1,xi));
            yi=max(0,min(shape[1]-1,yi));
            zi=max(0,min(shape[2]-1,zi));
            return arr[r3(xi,yi,zi)];
        }
        int xc=max(0,min(shape[0]-1,xi)),
            xn=(xi<0)?xc+1:(xi>=shape[0])?xc-1:xc;
        int yc=max(0,min(shape[1]-1,yi)),
            yn=(yi<0)?yc+1:(yi>=shape[1])?yc-1:yc;
        int zc=max(0,min(shape[2]-1,zi)),
            zn=(zi<0)?zc+1:(zi>=shape[2])?zc-1:zc;
        float v0=arr[r3(xc,yc,zc)], v1=arr[r3(xn,yn,zn)];
        return v0+(v0-v1);
    };
    float c00=fetch(x0,y0,z0)*(1-xd)+fetch(x1,y0,z0)*xd;
    float c01=fetch(x0,y0,z1)*(1-xd)+fetch(x1,y0,z1)*xd;
    float c10=fetch(x0,y1,z0)*(1-xd)+fetch(x1,y1,z0)*xd;
    float c11=fetch(x0,y1,z1)*(1-xd)+fetch(x1,y1,z1)*xd;
    float c0=c00*(1-yd)+c10*yd;
    float c1=c01*(1-yd)+c11*yd;
    return c0*(1-zd)+c1*zd;
}
extern "C" __global__ void warp_volume_kernel(
    const float* arr, const int* arr_shape,
    const float* disp0,const float* disp1,const float* disp2,
    const int* disp_shape,const float* scale,const float* offset,
    float* out,const int* out_shape)
{
    for(int i=blockIdx.x*blockDim.x+threadIdx.x; i<out_shape[0]; i+=blockDim.x*gridDim.x)
    for(int j=blockIdx.y*blockDim.y+threadIdx.y; j<out_shape[1]; j+=blockDim.y*gridDim.y)
    for(int k=blockIdx.z*blockDim.z+threadIdx.z; k<out_shape[2]; k+=blockDim.z*gridDim.z)
    {
        float x=(float)i/scale[0]+offset[0],
              y=(float)j/scale[1]+offset[1],
              z=(float)k/scale[2]+offset[2];
        float d0=trilinear_interp(disp0,disp_shape,x,y,z,EXTRAP_LINEAR);
        float d1=trilinear_interp(disp1,disp_shape,x,y,z,EXTRAP_LINEAR);
        float d2=trilinear_interp(disp2,disp_shape,x,y,z,EXTRAP_LINEAR);
        int idx = ravel3d(out_shape,i,j,k);
        out[idx]=trilinear_interp(
            arr,arr_shape,
            (float)i+d0,(float)j+d1,(float)k+d2,
            EXTRAP_ZERO
        );
    }
}
"""

_warp_volume_kernels = {}

def _get_warp_kernel(gpu_id):
    if gpu_id not in _warp_volume_kernels:
        with cp.cuda.Device(gpu_id):
            _warp_volume_kernels[gpu_id] = cp.RawKernel(
                _warp_volume_kernel_code, "warp_volume_kernel"
            )
    return _warp_volume_kernels[gpu_id]

def warp_volume(vol, disp_field, disp_scale, disp_offset,
                out=None, tpb=[8,8,8], gpu_id=0):
    with cp.cuda.Device(gpu_id):
        was_np = isinstance(vol, np.ndarray)
        vol    = cp.array(vol, dtype="float32", copy=False, order="C")
        df0    = cp.array(disp_field[0], dtype="float32", copy=False)
        df1    = cp.array(disp_field[1], dtype="float32", copy=False)
        df2    = cp.array(disp_field[2], dtype="float32", copy=False)
        scale  = cp.array(disp_scale, dtype="float32", copy=False)
        offset = cp.array(disp_offset, dtype="float32", copy=False)

        if out is None:
            out = cp.zeros(vol.shape, dtype="float32", order="C")
        else:
            out = cp.array(out, dtype="float32", copy=False, order="C")

        arr_shape  = cp.r_[vol.shape].astype("int32")
        disp_shape = cp.r_[disp_field.shape[1:]].astype("int32")
        out_shape  = cp.r_[out.shape].astype("int32")
        bpg = np.ceil(np.array(out.shape)/tpb).astype("int").tolist()

        kernel = _get_warp_kernel(gpu_id)
        kernel(
            tuple(bpg), tuple(tpb),
            (vol, arr_shape, df0, df1, df2,
             disp_shape, scale, offset,
             out, out_shape)
        )

        return cp.asnumpy(out) if was_np else out
