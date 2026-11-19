# Run: pytest tests/test_all.py
# All functions to be tested should start with test_prefix

import os
import subprocess

import pytest
import numpy as np
import h5py
import nibabel as nib
import tifffile

import warpfield
from warpfield.utils import load_data

from warpfield.backends import available_array_backends, backend_of, get_array_backend


def _to_numpy(x):
    return backend_of(x).to_numpy(x)


BACKENDS = available_array_backends()  # e.g. ["cupy", "numpy"], ["mlx", "numpy"] or ["numpy"]
GPU_AVAILABLE = len(BACKENDS) > 1
requires_gpu = pytest.mark.skipif(not GPU_AVAILABLE, reason="No GPU backend (CuPy with CUDA, or MLX) available.")


@pytest.fixture(params=BACKENDS)
def backend(request):
    return request.param


def test_trivial():
    assert True == True


def test_trivial2():
    assert False == False


def test_warpmap_invert_fast_zero_field(backend):
    """invert_fast should return (near-)zero inverse for a zero displacement field."""
    ref_shape = (5, 6, 7)
    mov_shape = (5, 6, 7)
    block_size = np.array([1, 1, 1], dtype=np.float32)
    block_stride = np.array([1, 1, 1], dtype=np.float32)
    warp_field = np.zeros((3, *ref_shape), dtype=np.float32)

    wm = warpfield.register.WarpMap(
        warp_field=warp_field,
        block_size=block_size,
        block_stride=block_stride,
        ref_shape=ref_shape,
        mov_shape=mov_shape,
        backend=backend,
    )

    inv = wm.invert_fast(sigma=0.5, truncate=4)
    inv_np = _to_numpy(inv.warp_field)

    expected_shape = tuple(np.ceil(np.array(mov_shape) / block_stride + 1).astype(int))
    assert inv_np.shape == (3, *expected_shape)
    np.testing.assert_allclose(inv_np, 0.0, atol=1e-6)


def test_warpmap_push_coordinates_constant_shift(backend):
    """push_coordinates should add sampled displacement vectors to input voxel coordinates."""
    shape = (6, 6, 6)
    shift = np.array([1.5, -2.0, 0.25], dtype=np.float32)
    warp_field = np.zeros((3, *shape), dtype=np.float32)
    warp_field[0] = shift[0]
    warp_field[1] = shift[1]
    warp_field[2] = shift[2]

    wm = warpfield.register.WarpMap(
        warp_field=warp_field,
        block_size=np.array([1, 1, 1], dtype=np.float32),
        block_stride=np.array([1, 1, 1], dtype=np.float32),
        ref_shape=shape,
        mov_shape=shape,
        backend=backend,
    )

    coords = np.array([[1.0, 2.0, 3.5], [1.0, 4.0, 2.0], [2.0, 1.5, 4.0]], dtype=np.float32)

    pushed = wm.push_coordinates(coords)
    pushed_negative = wm.push_coordinates(coords, negative_shifts=True)

    np.testing.assert_allclose(pushed, coords + shift[:, None], atol=1e-5)
    np.testing.assert_allclose(pushed_negative, coords - shift[:, None], atol=1e-5)


def test_warpmap_jacobian_det_linear_field(backend):
    """jacobian_det should match the analytic determinant for a linear displacement field."""
    shape = (7, 7, 7)
    z, y, x = np.indices(shape, dtype=np.float32)
    alpha, beta, gamma = 0.2, -0.1, 0.05
    warp_field = np.stack([alpha * z, beta * y, gamma * x]).astype(np.float32)

    wm = warpfield.register.WarpMap(
        warp_field=warp_field,
        block_size=np.array([1, 1, 1], dtype=np.float32),
        block_stride=np.array([1, 1, 1], dtype=np.float32),
        ref_shape=shape,
        mov_shape=shape,
        backend=backend,
    )

    det_j = _to_numpy(wm.jacobian_det(units_per_voxel=[1, 1, 1], edge_order=1))
    expected = (1.0 + alpha) * (1.0 + beta) * (1.0 + gamma)

    np.testing.assert_allclose(det_j[1:-1, 1:-1, 1:-1], expected, atol=2e-3, rtol=2e-3)


def test_import_npy(tmp_path):
    """Test importing a 3D .npy file."""
    file_path = tmp_path / "test.npy"
    data = np.random.rand(10, 10, 10).astype("float32")
    np.save(file_path, data)

    loaded_data, meta = load_data(str(file_path))
    assert np.allclose(loaded_data, data), "Loaded .npy data does not match expected data."


def test_import_h5(tmp_path):
    """Test importing a 3D .h5 file."""
    file_path = tmp_path / "test.h5"
    data = np.random.rand(10, 10, 10).astype("float32")
    with h5py.File(file_path, "w") as f:
        f.create_dataset("dataset", data=data)

    loaded_data, meta = load_data(f"{file_path}:dataset")
    assert np.allclose(loaded_data, data), "Loaded .h5 data does not match expected data."


def test_import_nii(tmp_path):
    """Test importing a 3D .nii file."""
    file_path = tmp_path / "test.nii"
    data = np.random.rand(10, 10, 10).astype("float32")
    nii = nib.Nifti1Image(data, affine=np.eye(4))
    nib.save(nii, file_path)

    loaded_data, meta = load_data(str(file_path))
    assert np.allclose(loaded_data, data), "Loaded .nii data does not match expected data."


def test_import_tiff(tmp_path):
    """Test importing a 3D .tiff file."""
    file_path = tmp_path / "test.tiff"
    data = np.random.rand(10, 10, 10).astype("float32")
    tifffile.imwrite(file_path, data)

    loaded_data, meta = load_data(str(file_path))
    assert np.allclose(loaded_data, data), "Loaded .tiff data does not match expected data."


@requires_gpu
@pytest.mark.parametrize("shape", [(256, 256, 256), (256, 256, 180)])  # non-cubic catches axis mix-ups
def test_register_volumes(shape):
    """Test the register_volumes function."""
    import scipy.ndimage

    fixed = scipy.ndimage.gaussian_filter(np.random.rand(*shape).astype("float32"), sigma=1)
    moving = np.roll(fixed, shift=5, axis=0).copy()  # Simulate a simple shift
    recipe = warpfield.Recipe.from_yaml("default.yml")

    mlx_used = get_array_backend("auto").name == "mlx"
    if mlx_used:
        import mlx.core as mx

        mx.reset_peak_memory()

    registered, warp_map, _ = warpfield.register_volumes(fixed, moving, recipe, backend="auto", verbose=False)

    assert registered.shape == fixed.shape, "Registered volume shape mismatch."
    assert (
        np.abs(registered[10:-10, 10:-10, 10:-10] - fixed[10:-10, 10:-10, 10:-10]) < 0.2
    ).mean() > 0.9, "Registered volume does not match the fixed volume."
    assert warp_map is not None, "WarpMap object was not returned."

    if mlx_used:
        peak = mx.get_peak_memory() / 1024**3  # in GiB
        assert peak < 2.0, f"MLX peak memory usage is too high: {peak:.2f} GiB"


@requires_gpu
def test_cli(tmp_path):
    """Test the CLI for registering volumes."""
    fixed = np.random.rand(256, 256, 256).astype("float32")
    moving = np.roll(fixed, shift=5, axis=0).copy()  # Simulate a simple shift

    # Save mock volumes to temporary files
    fixed_path = tmp_path / "fixed.npy"
    moving_path = tmp_path / "moving.npy"
    np.save(fixed_path, fixed)
    np.save(moving_path, moving)

    # Path to the default recipe
    recipe_path = "default.yml"

    # Output path
    output_path = tmp_path / "output.h5"

    # Run the CLI command
    result = subprocess.run(
        [
            "python",
            "-m",
            "warpfield",
            "--fixed",
            str(fixed_path),
            "--moving",
            str(moving_path),
            "--recipe",
            recipe_path,
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    # Check the CLI execution
    assert result.returncode == 0, f"CLI failed with error: {result.stderr}"

    # Verify the output file
    assert os.path.exists(output_path), "Output file was not created."
    import h5py

    with h5py.File(output_path, "r") as f:
        assert "moving_reg" in f, "Registered volume not found in output file."
        assert "warp_map" in f, "Warp map not found in output file."

def _small_volumes(shift=3, size=64):
    import scipy.ndimage

    fixed = scipy.ndimage.gaussian_filter(np.random.default_rng(0).random((size,) * 3).astype("float32"), 2)
    return fixed, np.roll(fixed, shift, axis=0)


def test_register_volumes_small(backend):
    """A short registration runs on every backend (incl. NumPy on machines without a GPU)."""
    fixed, moving = _small_volumes()
    recipe = warpfield.Recipe()
    recipe.add_level(block_size=[32, 32, 32], repeats=2)
    registered, warp_map, _ = warpfield.register_volumes(fixed, moving, recipe, backend=backend, verbose=False)

    assert isinstance(registered, np.ndarray) and registered.shape == fixed.shape
    assert isinstance(warp_map, warpfield.WarpMap)
    assert warp_map.B.name == backend
    np.testing.assert_allclose(_to_numpy(warp_map.warp_field)[0].mean(), 3.0, atol=0.1)
    assert (np.abs(registered[8:-8, 8:-8, 8:-8] - fixed[8:-8, 8:-8, 8:-8]) < 0.02).mean() > 0.99


def test_custom_projector_callable(backend):
    """User callables for `project` take (vol_blocks, axis), as documented."""
    fixed, moving = _small_volumes()
    recipe = warpfield.Recipe()
    recipe.levels = recipe.levels[:1]
    calls = []

    def project(vol_blocks, axis):
        calls.append(axis)
        return warpfield.register.Projector()(vol_blocks, axis)

    recipe.levels[0].project = project
    _, warp_map, _ = warpfield.register_volumes(fixed, moving, recipe, backend=backend, verbose=False)
    assert len(calls) > 0
    np.testing.assert_allclose(_to_numpy(warp_map.warp_field)[0].mean(), 3.0, atol=0.1)


def test_warpmap_h5_roundtrip(backend, tmp_path):
    rng = np.random.default_rng(0)
    wm = warpfield.WarpMap(rng.random((3, 4, 5, 6)), [8, 8, 8], [4, 4, 4], (20, 24, 28), (21, 25, 29), backend=backend)
    wm.to_h5(tmp_path / "wm.h5")
    wm2 = warpfield.WarpMap.from_h5(tmp_path / "wm.h5", backend=backend)
    assert isinstance(wm2, warpfield.WarpMap) and wm2.B.name == backend
    np.testing.assert_allclose(_to_numpy(wm2.warp_field), _to_numpy(wm.warp_field))
    assert (wm2.ref_shape, wm2.mov_shape) == (wm.ref_shape, wm.mov_shape)


def test_push_pull_coordinates_roundtrip(backend):
    """Pulling pushed coordinates should return the original coordinates."""
    ref_shape = (128, 128, 128)
    block_size = np.array([16, 16, 16])
    grid = np.indices((8, 8, 8)).astype("float32")
    warp_field = np.stack([3.0 + np.sin(grid[1] / 3), -2.0 + np.cos(grid[2] / 4), 2.5 * np.ones_like(grid[0])])
    warp_map = warpfield.WarpMap(warp_field, block_size, block_size, ref_shape, ref_shape, backend=backend)

    points = np.random.default_rng(0).uniform(32, 96, size=(3, 200)).astype("float32")
    pulled = warp_map.pull_coordinates(warp_map.push_coordinates(points))
    assert isinstance(pulled, np.ndarray)
    assert np.abs(pulled - points).max() < 0.25, "pull_coordinates does not reverse push_coordinates."


def test_smoother_shear(backend):
    """Smoother with shear should run, and shear=0 should match the non-shear path."""
    B = get_array_backend(backend)
    x = np.random.default_rng(0).standard_normal((6, 6, 6, 9, 9)).astype("float32")
    sigmas, block_size = [1.0, 1.5, 0.8], np.array([32, 32, 32])
    Smoother = warpfield.register.Smoother

    plain = _to_numpy(Smoother(sigmas=sigmas)(B.asarray(x), block_size))
    sheared0 = _to_numpy(Smoother(sigmas=sigmas, shear=0.0)(B.asarray(x), block_size))
    sheared = _to_numpy(Smoother(sigmas=sigmas, shear=0.5)(B.asarray(x), block_size))
    np.testing.assert_allclose(sheared0, plain, atol=1e-5)
    assert not np.allclose(sheared, plain, atol=1e-3)


def test_callbacks(backend):
    """mips_callback and mosaic_callback accept arrays of every backend and return 2D NumPy images."""
    pytest.importorskip("skimage")
    B = get_array_backend(backend)
    vol = B.asarray(np.random.default_rng(0).random((20, 30, 40)), B.float32)
    for cb in [warpfield.utils.mips_callback(), warpfield.utils.mosaic_callback(num_slices=4, thick=2)]:
        out = cb(vol)
        assert isinstance(out, np.ndarray) and out.ndim == 2


def test_numpy_backend_needs_no_gpu(monkeypatch):
    """backend="numpy" must never fall back to the default GPU backend (as on CI runners without a GPU)."""
    from warpfield.backends import selection

    monkeypatch.setitem(selection._USABLE, "cupy", lambda: False)
    monkeypatch.setitem(selection._USABLE, "mlx", lambda: False)
    monkeypatch.setattr(selection, "_default", None)
    with pytest.raises(RuntimeError):
        get_array_backend("auto")  # no GPU backend left
    fixed, moving = _small_volumes()
    recipe = warpfield.Recipe()
    recipe.levels = recipe.levels[:1]
    registered, warp_map, _ = warpfield.register_volumes(fixed, moving, recipe, backend="numpy", verbose=False)
    assert isinstance(registered, np.ndarray) and warp_map.B.name == "numpy"


def test_set_default_backend_normalises_name(backend):
    try:
        warpfield.set_default_backend(backend.upper())
        assert get_array_backend("auto").name == backend
    finally:
        warpfield.set_default_backend("auto")


def test_warpmap_assignment_converts_to_backend(backend):
    B = get_array_backend(backend)
    wm = warpfield.WarpMap(np.zeros((3, 2, 2, 2)), [1, 1, 1], [1, 1, 1], (2, 2, 2), (2, 2, 2), backend=backend)
    wm.warp_field = np.ones((3, 2, 2, 2))  # float64 NumPy array
    assert B.is_array(wm.warp_field) and wm.warp_field.dtype == B.float32
    wm.warp_field *= 2
    assert B.is_array(wm.warp_field)


def test_fit_affine_single_block_axis(backend):
    """An affine fit with only one block along an axis (rank-deficient system) must still work."""
    shift = np.array([2.0, -1.0, 0.5], dtype="float32")
    warp_field = np.broadcast_to(shift[:, None, None, None], (3, 1, 6, 6)).copy()
    wm = warpfield.WarpMap(warp_field, [32, 32, 32], [32, 32, 32], (32, 192, 192), (32, 192, 192), backend=backend)
    fitted, _ = wm.fit_affine()
    np.testing.assert_allclose(_to_numpy(fitted.warp_field), warp_field, atol=1e-3)


def test_copilot_review_cases(backend):
    """Cases from the Copilot review of #48."""
    B = get_array_backend(backend)
    # soften_edges on a 2D image with the default soft_edge
    img = np.random.default_rng(0).random((20, 30)).astype("float32")
    np.testing.assert_allclose(warpfield.ndimage.soften_edges(img, backend=backend), img)
    # WarpMap does not share memory with the arrays it was created from
    field = np.zeros((3, 2, 2, 2), dtype="float32")
    wm = warpfield.WarpMap(B.asarray(field), [1, 1, 1], [1, 1, 1], (2, 2, 2), (2, 2, 2), backend=backend)
    src = wm.warp_field
    wm2 = warpfield.WarpMap(src, [1, 1, 1], [1, 1, 1], (2, 2, 2), (2, 2, 2), backend=backend)
    wm2.warp_field += 1
    assert _to_numpy(wm.warp_field).max() == 0
    # chain works across backends (NumPy map + this backend's map)
    wm_np = warpfield.WarpMap(np.ones_like(field), [1, 1, 1], [1, 1, 1], (2, 2, 2), (2, 2, 2), backend="numpy")
    assert _to_numpy(wm.chain(wm_np).warp_field).min() == 1
    # mosaic_callback with slabs reaching past the last slice
    pytest.importorskip("skimage")
    vol = B.asarray(np.random.default_rng(0).random((10, 12, 14)), B.float32)
    assert warpfield.utils.mosaic_callback(num_slices=9, thick=10)(vol).ndim == 2


# ------------------------------------------------------------- cases from the maintainer's review of #48
def test_custom_callables_may_return_numpy(backend):
    """A custom pre_filter / projector / smoother that returns NumPy must not move the pipeline to the host."""
    import scipy.ndimage

    fixed, moving = _small_volumes()
    recipe = warpfield.Recipe()
    recipe.levels = recipe.levels[:1]
    recipe.pre_filter = lambda vol, reg_mask=None: scipy.ndimage.gaussian_filter(_to_numpy(vol), 1.0)
    proj = warpfield.register.Projector()
    recipe.levels[0].project = lambda blocks, axis: _to_numpy(proj(blocks, axis))
    smooth = recipe.levels[0].smooth
    recipe.levels[0].smooth = lambda xcorr, block_size: _to_numpy(smooth(xcorr, block_size))
    registered, warp_map, _ = warpfield.register_volumes(fixed, moving, recipe, backend=backend, verbose=False)
    assert isinstance(registered, np.ndarray) and warp_map.B.name == backend


def test_zoom_chop_pad_matches_reference(backend):
    """zoom_chop_pad with NumPy scale/shape inputs (as match_volumes passes them) matches the formula on main."""
    import scipy.ndimage

    x = np.random.default_rng(0).random((10, 12, 14)).astype("float32")
    target_shape, scale = np.array([13, 9, 16]), np.array([1.3, 0.8, 1.1])
    shift, flip = np.array([0.5, -1.0, 2.0]), (False, True, False)
    out = warpfield.ndimage.zoom_chop_pad(x, target_shape, scale, shift=shift, flip=flip, backend=backend)
    coords = np.indices(target_shape).astype("float64")
    for i in range(3):
        coords[i] = (coords[i] - target_shape[i] / 2) / scale[i] + x.shape[i] / 2
        if flip[i]:
            coords[i] = -coords[i] + x.shape[i] - 1
        coords[i] -= shift[i]
    expected = scipy.ndimage.map_coordinates(x, coords, order=1, mode="constant", cval=0)
    assert isinstance(out, np.ndarray) and out.dtype == np.float32
    np.testing.assert_allclose(out, expected, atol=1e-5)


def test_warpmap_pickle_and_deepcopy(backend):
    import copy
    import pickle

    field = np.random.default_rng(0).random((3, 2, 3, 4)).astype("float32")
    wm = warpfield.WarpMap(field, [4, 4, 4], [2, 2, 2], (4, 6, 8), (4, 6, 8), backend=backend)
    for wm2 in (pickle.loads(pickle.dumps(wm)), copy.deepcopy(wm)):
        assert wm2.B is wm.B
        np.testing.assert_array_equal(_to_numpy(wm2.warp_field), field)
        wm2.warp_field += 1
        np.testing.assert_array_equal(_to_numpy(wm.warp_field), field)


def test_warpmap_copies_backend_inputs(backend):
    """WarpMap owns its arrays, also when given arrays already on its backend (CuPy returns views there)."""
    B = get_array_backend(backend)
    f = B.asarray(np.zeros((3, 2, 2, 2), "float32"))
    bs = B.asarray(np.ones(3, "float32"))
    wm = warpfield.WarpMap(f, bs, bs, (2, 2, 2), (2, 2, 2), backend=backend)
    wm.warp_field += 1
    wm.block_size *= 2
    assert _to_numpy(f).sum() == 0 and _to_numpy(bs).sum() == 3
    wm2 = wm.median_filter()  # made via _new
    wm2.block_stride *= 3
    assert _to_numpy(wm.block_stride).sum() == 3
    wm2.warp_field = wm.warp_field  # assignment copies too
    wm2.warp_field *= 0
    assert _to_numpy(wm.warp_field).sum() == 24


def test_soften_edges_copy_keeps_memmap(tmp_path):
    path = tmp_path / "vol.npy"
    np.save(path, np.ones((8, 9, 10), "float32"))
    mm = np.load(path, mmap_mode="r+")
    out = warpfield.ndimage.soften_edges(mm, soft_edge=2, copy=True, backend="numpy")
    assert out[0, 0, 0] == 0
    mm.flush()
    assert np.load(path).min() == 1


def test_backend_of_without_usable_mlx(monkeypatch):
    """MLX arrays can be converted even where the MLX backend cannot register (MLX on CPU or CUDA)."""
    mx = pytest.importorskip("mlx.core")
    from warpfield.backends import selection

    monkeypatch.setitem(selection._USABLE, "mlx", lambda: False)
    monkeypatch.setattr(selection, "_instances", {})
    a = mx.ones((2, 3))
    assert backend_of(a).name == "mlx"
    np.testing.assert_array_equal(backend_of(a).to_numpy(a), np.ones((2, 3)))
    np.testing.assert_array_equal(get_array_backend("numpy").asarray(a), np.ones((2, 3)))
    with pytest.raises(RuntimeError):
        get_array_backend("mlx")  # asking for it by name still checks
    with pytest.raises(RuntimeError):
        warpfield.WarpMap(mx.zeros((3, 2, 2, 2)), [1] * 3, [1] * 3, (2,) * 3, (2,) * 3)  # so does computing on it


def test_soften_edges_copy_keeps_input(backend):
    """copy=True never changes the input, whatever its type (an MLX array viewed by NumPy shares memory)."""
    inputs = [np.ones((6, 7, 8), "float32"), np.ones((6, 7, 8))]
    if backend != "numpy":
        inputs.append(get_array_backend(backend).asarray(np.ones((6, 7, 8), "float32")))
    for arr in inputs:
        for b in {backend, "numpy"}:
            warpfield.ndimage.soften_edges(arr, soft_edge=2, copy=True, backend=b)
            assert _to_numpy(arr).min() == 1, (type(arr), b)


def test_copies_with_own_backend_object():
    """Passing a new backend object (not the cached instance) must not skip the copies."""
    from warpfield.backends.numpy_backend import NumpyBackend

    field = np.zeros((3, 2, 2, 2), "float32")
    wm = warpfield.WarpMap(field, [1] * 3, [1] * 3, (2,) * 3, (2,) * 3, backend=NumpyBackend())
    wm.warp_field += 1
    vol = np.ones((6, 7, 8), "float32")
    warpfield.ndimage.soften_edges(vol, soft_edge=2, copy=True, backend=NumpyBackend())
    assert field.sum() == 0 and vol.min() == 1
