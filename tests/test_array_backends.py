"""Tests for the array backend layer (warpfield.backends).

Every method of every backend that can run on this machine is compared with NumPy/SciPy. The NumPy
backend's port of the CUDA warp kernel is compared with output saved from CuPy.
"""

from pathlib import Path

import numpy as np
import pytest
import scipy.ndimage
import scipy.signal.windows

from warpfield.backends import COMMON_XP, available_array_backends, backend_of, get_array_backend

ALL_BACKENDS = available_array_backends()
GPU_BACKENDS = [name for name in ALL_BACKENDS if name != "numpy"]
WARP_DATA = Path(__file__).parent / "data" / "cupy_groundtruth_warp"


@pytest.fixture(params=ALL_BACKENDS)
def B(request):
    return get_array_backend(request.param)


@pytest.fixture(params=GPU_BACKENDS)
def G(request):
    return get_array_backend(request.param)


def rand(*shape, seed=0):
    return np.random.default_rng(seed).standard_normal(shape).astype("float32")


def check(B, out, expected, atol=1e-5, rtol=1e-5):
    assert B.is_array(out), f"expected a {B.name} array, got {type(out)}"
    np.testing.assert_allclose(B.to_numpy(out), expected, atol=atol, rtol=rtol)


# ------------------------------------------------------------------------- basics
def test_common_xp_names_exist(B):
    for name in COMMON_XP:
        obj = B.xp
        for part in name.split("."):
            assert hasattr(obj, part), f"{B.name}: xp.{name} is missing"
            obj = getattr(obj, part)


def test_auto_selection_only_picks_gpu():
    if GPU_BACKENDS:
        assert get_array_backend("auto").name in GPU_BACKENDS
    else:
        with pytest.raises(RuntimeError):
            get_array_backend("auto")


def test_conversion_and_copy(B):
    a = np.random.default_rng(0).random((4, 5, 6))  # float64
    x = B.asarray(a, "float32")
    assert B.is_array(x) and backend_of(x) is B
    assert B.to_numpy(x).dtype == np.float32
    np.testing.assert_allclose(B.to_numpy(x), a, rtol=1e-6)
    assert B.asarray(x) is x or B.name == "numpy"  # no copy for arrays already on the backend
    c = B.copy(x)
    c[0] = 0
    assert np.all(B.to_numpy(x)[0] != 0), "copy shares memory with the original"


def test_indices(B):
    check(B, B.indices((3, 4, 5), B.float32), np.indices((3, 4, 5)), atol=0)


def test_argmax_2d(B):
    a = rand(3, 4, 6, 7)
    i, j = B.argmax_2d(B.asarray(a))
    ei, ej = np.unravel_index(a.reshape(3, 4, -1).argmax(-1), a.shape[-2:])
    check(B, i, ei, atol=0)
    check(B, j, ej, atol=0)
    check(B, i - 10, ei - 10, atol=0)  # must be signed


# ----------------------------------------------------------------- linear algebra
def test_lstsq(B):
    a = np.c_[rand(50, 3) * 10, np.ones(50, dtype="float32")]
    b = rand(50, 3, seed=1)
    check(B, B.lstsq(B.asarray(a), B.asarray(b)), np.linalg.lstsq(a, b, rcond=None)[0], atol=1e-4)


def test_det(B):
    a = rand(5, 6, 3, 3)
    check(B, B.det(B.asarray(a)), np.linalg.det(a), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("edge_order", [1, 2])
def test_gradient(B, edge_order):
    f = rand(6, 7, 8)
    out = B.gradient(B.asarray(f), 1.0, 2.0, 0.5, edge_order=edge_order)
    for o, e in zip(out, np.gradient(f, 1.0, 2.0, 0.5, edge_order=edge_order)):
        check(B, o, e, atol=1e-5)


# ------------------------------------------------------------------------ ndimage
@pytest.mark.parametrize(
    "shape,sigma,mode,truncate",
    [
        ((12, 14, 16), 1.5, "reflect", 4.0),  # RegFilter DoG on a volume
        ((4, 5, 3, 9, 11), [0, 0, 0, 0.5, 2.0], "reflect", 5.0),  # Projector on block projections
        ((4, 5, 3, 9, 11), [1.0, 1.5, 0.8, 0, 0], "constant", 4.0),  # Smoother on cross-correlations
    ],
)
def test_gaussian_filter(B, shape, sigma, mode, truncate):
    x = rand(*shape)
    expected = scipy.ndimage.gaussian_filter(x, sigma, mode=mode, truncate=truncate)
    check(B, B.gaussian_filter(B.asarray(x), sigma, mode=mode, truncate=truncate), expected)


@pytest.mark.parametrize("kernel_shape,mode", [((5, 7), "constant"), ((5, 7), "reflect"), ((3, 5, 3), "reflect")])
def test_convolve_leading(B, kernel_shape, mode):
    x, k = rand(6, 6, 5, 9, 9), np.abs(rand(*kernel_shape, seed=1))
    expected = scipy.ndimage.convolve(x, k.reshape(k.shape + (1,) * (x.ndim - k.ndim)), mode=mode)
    check(B, B.convolve_leading(B.asarray(x), B.asarray(k), mode=mode), expected, atol=1e-4)


@pytest.mark.parametrize("size", [[1, 3, 3, 3], [1, 5, 5, 5]])
def test_median_filter(B, size):
    x = rand(3, 8, 9, 10)
    expected = scipy.ndimage.median_filter(x, size=size, mode="nearest")
    check(B, B.median_filter(B.asarray(x), size), expected, atol=0)


@pytest.mark.parametrize("order", [0, 1])
def test_map_coordinates_nearest(B, order):
    x = rand(8, 9, 10)
    coords = np.random.default_rng(1).uniform(-2, 12, size=(3, 500)).astype("float32")
    expected = scipy.ndimage.map_coordinates(x, coords, order=order, mode="nearest")
    check(B, B.map_coordinates(B.asarray(x), B.asarray(coords), order=order), expected)


def test_map_coordinates_constant(B):
    x = rand(8, 9, 10)
    coords = np.random.default_rng(1).uniform(-2, 12, size=(3, 4, 5, 6)).astype("float32")
    expected = scipy.ndimage.map_coordinates(x, coords, order=1, mode="constant", cval=0.0)
    check(B, B.map_coordinates(B.asarray(x), B.asarray(coords), mode="constant"), expected)


def test_zoom(B):
    x = rand(8, 9, 10)
    check(B, B.zoom(B.asarray(x), (0.5, 1.3, 2.0)), scipy.ndimage.zoom(x, (0.5, 1.3, 2.0), order=1))


@pytest.mark.parametrize("alpha", [0.0, 0.3, 1.0])
def test_tukey(B, alpha):
    check(B, B.tukey(17, alpha), scipy.signal.windows.tukey(17, alpha))


def test_sliding_block(B):
    x = rand(12, 14, 16)
    expected = np.lib.stride_tricks.sliding_window_view(x, (4, 5, 6))[::2, ::3, ::4]
    check(B, B.sliding_block(B.asarray(x), (4, 5, 6), (2, 3, 4)), expected, atol=0)


# ---------------------------------------------------------------------------- FFT
@pytest.mark.parametrize("shape", [(2, 3, 8, 10), (2, 3, 7, 9)])
def test_fft_roundtrip(B, shape):
    x = rand(*shape)
    X = B.rfft2(B.asarray(x))
    check(B, X, np.fft.rfftn(x, axes=(-2, -1)), atol=1e-4)
    check(B, B.irfft2(X, s=shape[-2:]), x, atol=1e-5)
    if shape[-1] % 2 == 0:
        check(B, B.irfft2(X), x, atol=1e-5)


# -------------------------------------------------------------------- warp kernel
def test_warp_volume_out_buffer(B):
    """`out` sets the output shape, is filled and returned."""
    vol = rand(9, 10, 11)
    out = B.asarray(np.full((8, 10, 11), -999, dtype="float32"))
    res = B.warp_volume(B.asarray(vol), B.asarray(np.zeros((3, 9, 10, 11), "float32")), [1, 1, 1], [0, 0, 0], out=out)
    assert res is out
    check(B, out, vol[:8])


def test_warp_volume_matches_numpy(G):
    vol, disp = rand(19, 23, 27), 2 * rand(3, 7, 8, 9, seed=1)
    scale, offset = np.array([3.0, 3.0, 3.0], "float32"), np.array([-0.5, 0.2, -1.0], "float32")
    expected = get_array_backend("numpy").warp_volume(vol, disp, scale, offset, out=np.zeros((20, 22, 25), "float32"))
    out = G.warp_volume(G.asarray(vol), G.asarray(disp), scale, offset, out=G.asarray(np.zeros((20, 22, 25), "float32")))
    check(G, out, expected, atol=1e-4)


def test_warp_volume_single_point_field(B):
    """A field with one point per axis (translation level) is extrapolated as a constant shift.

    The kernels used to read past the end of the field here.
    """
    vol = rand(20, 22, 24)
    disp = np.float32([2.5, -1.0, 0.75]).reshape(3, 1, 1, 1)
    out = B.warp_volume(B.asarray(vol), B.asarray(disp), np.float32([20, 22, 24]), np.float32([-0.5, -0.5, -0.5]))
    expected = scipy.ndimage.map_coordinates(
        vol, np.indices(vol.shape) + disp.reshape(3, 1, 1, 1), order=1, mode="grid-constant", cval=0
    )  # the kernel interpolates towards 0 beyond the last voxel, as "grid-constant" does
    check(B, out, expected, atol=1e-5)


@pytest.mark.parametrize(
    "case,scale,offset,out_shape",
    [
        ("identity_numpy", [1, 1, 1], [0, 0, 0], None),
        ("const_shift", [1, 1, 1], [0, 0, 0], None),
        ("random_samegrid", [1, 1, 1], [0, 0, 0], None),
        ("random_scaledgrid", [3.0, 2.5, 2.0], [-0.2, 0.1, -0.3], (18, 26, 34)),
        ("numpy_mixed", [1, 1, 1], [0, 0, 0], None),
    ],
)
def test_numpy_warp_matches_cupy_output(case, scale, offset, out_shape):
    if not (WARP_DATA / f"warp_{case}_output.npy").exists():
        pytest.skip("CuPy reference data not available")
    vol = np.load(WARP_DATA / f"warp_{case}_input.npy")
    disp = np.load(WARP_DATA / f"warp_{case}_disp.npy")
    out = None if out_shape is None else np.zeros(out_shape, "float32")
    res = get_array_backend("numpy").warp_volume(vol, disp, np.float32(scale), np.float32(offset), out=out)
    np.testing.assert_allclose(res, np.load(WARP_DATA / f"warp_{case}_output.npy"), atol=1e-5, rtol=1e-5)


def test_gaussian_filter_other_modes(B):
    x = rand(6, 7, 8)
    check(B, B.gaussian_filter(B.asarray(x), 1.5, mode="nearest"), scipy.ndimage.gaussian_filter(x, 1.5, mode="nearest"))
