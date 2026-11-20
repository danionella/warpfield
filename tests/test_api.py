"""Tests for the public API around the registration: recipes, CLI, WarpMap helpers and array types."""

import os
import subprocess
import sys

import numpy as np
import pytest
import h5py

import warpfield
from warpfield.__main__ import _check_output_path
from warpfield.backends import available_array_backends, backend_of

BACKENDS = available_array_backends()
requires_cupy = pytest.mark.skipif("cupy" not in BACKENDS, reason="CuPy with CUDA not available.")


@pytest.fixture(params=BACKENDS)
def backend(request):
    """Runs the test on each available backend (set for the whole process, as users do)."""
    warpfield.set_backend(request.param)
    yield request.param
    warpfield.set_backend("auto")


def _to_numpy(x):
    return backend_of(x).to_numpy(x)


def _small_volumes(shift=3, size=64):
    import scipy.ndimage

    fixed = scipy.ndimage.gaussian_filter(np.random.default_rng(0).random((size,) * 3).astype("float32"), 2)
    return fixed, np.roll(fixed, shift, axis=0)


# ---------------------------------------------------------------------------------------------------- recipes
@pytest.mark.parametrize("name", ["default.yml", "default_opm.yml"])
def test_bundled_recipes_load(name):
    recipe = warpfield.Recipe.from_yaml(name)
    assert len(recipe.levels) > 2


@pytest.mark.parametrize(
    "data",
    [
        {"levels": [{"block_size": [8, 8, 8], "blocksize": [4, 4, 4]}]},  # misspelled level key
        {"levels": [{"block_size": [8, 8, 8], "shear": 0.5}]},  # key at the wrong level (belongs in smooth)
        {"levels": [{"block_size": [8, 8, 8], "smooth": {"sigma": [5, 5, 5]}}]},  # misspelled nested key
        {"pre_filter": {"clip_tresh": 10}},
    ],
)
def test_recipe_rejects_unknown_keys(data):
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        warpfield.Recipe.model_validate(data)


def test_recipe_with_subclassed_pieces(backend):
    """Subclasses of the recipe pieces may add their own settings, __init__ and validators (as on 0.2):
    extra="forbid" is for parsing, and the check before a registration only looks at warpfield's settings."""
    from pydantic import field_validator

    class MyProjector(warpfield.register.Projector):
        gamma: float = 1.0

    class InitProjector(warpfield.register.Projector):
        gamma: float = 1.0

        def __init__(self, g, **kwargs):
            super().__init__(gamma=g, **kwargs)

    class MyFilter(warpfield.register.RegFilter):
        kernel: str = "3"

        @field_validator("kernel")
        @classmethod
        def to_array(cls, v):
            return np.ones(int(v))  # changes the type: re-validating as MyFilter would fail

    fixed, moving = _small_volumes()
    for projector in [MyProjector(gamma=2.0), InitProjector(2.0)]:
        recipe = warpfield.Recipe(pre_filter=MyFilter(kernel="5"))
        recipe.levels = recipe.levels[:1]
        recipe.levels[0].project = projector
        _, warp_map, _ = warpfield.register_volumes(fixed, moving, recipe, verbose=False)
        np.testing.assert_allclose(_to_numpy(warp_map.warp_field)[0].mean(), 3.0, atol=0.1)


@pytest.mark.parametrize(
    "change",
    [
        lambda r: setattr(r.levels[0], "repeats", "many"),
        lambda r: r.levels[0].smooth.sigmas.__setitem__(0, "wide"),  # in-place change to a nested list
        lambda r: setattr(r.levels[0].project, "low", "x"),
    ],
)
def test_recipe_invalid_changes_are_caught(change):
    """Invalid values set after the recipe was created are caught before the registration starts, and the error
    says where."""
    fixed, moving = _small_volumes(size=32)
    recipe = warpfield.Recipe()
    change(recipe)
    with pytest.raises(ValueError, match=r"Invalid setting in recipe\.levels\[0\]"):
        warpfield.register_volumes(fixed, moving, recipe, verbose=False)


def test_recipe_yaml_roundtrip(tmp_path):
    recipe = warpfield.Recipe.from_yaml("default.yml")
    recipe.to_yaml(tmp_path / "r.yml")
    assert warpfield.Recipe.from_yaml(str(tmp_path / "r.yml")) == recipe


# ------------------------------------------------------------------------------------------------ WarpMap
@pytest.mark.parametrize(
    "stride, units", [([1, 1, 1], [1, 1, 1]), ([4, 2, 4], [1, 1, 1]), ([2, 2, 2], [1, 1, 4]), ([4, 4, 4], [2, 3, 2])]
)
def test_jacobian_det_with_stride_and_units(backend, stride, units):
    """A displacement u = a * x (in voxels) has det J = prod(1 + a), whatever the grid stride and voxel size."""
    a = np.array([0.2, -0.1, 0.05])
    shape = (7, 8, 9)
    idx = np.indices(shape, dtype=np.float32)
    warp_field = np.stack([a[i] * idx[i] * stride[i] for i in range(3)]).astype("float32")  # in voxels
    wm = warpfield.WarpMap(warp_field, stride, stride, (32,) * 3, (32,) * 3)
    det = _to_numpy(wm.jacobian_det(units_per_voxel=units))
    np.testing.assert_allclose(det[1:-1, 1:-1, 1:-1], np.prod(1 + a), rtol=1e-4)


def test_projector_callable_called_with_axis_keyword(backend):
    """Custom projectors are called as f(vol_blocks, axis=axis), so keyword-only signatures work too."""
    fixed, moving = _small_volumes()
    recipe = warpfield.Recipe()
    recipe.levels = recipe.levels[:1]
    recipe.levels[0].project = lambda vol_blocks, *, axis: warpfield.register.Projector()(vol_blocks, axis)
    _, warp_map, _ = warpfield.register_volumes(fixed, moving, recipe, verbose=False)
    np.testing.assert_allclose(_to_numpy(warp_map.warp_field)[0].mean(), 3.0, atol=0.1)


def test_recipe_pieces_run_on_current_backend(backend, monkeypatch):
    """RegFilter, Projector and Smoother compute on the backend warpfield runs on, also for NumPy input (as in
    0.2, recipe.pre_filter(vol) uses the GPU), and return NumPy for NumPy input."""
    from warpfield.backends import get_array_backend

    B = get_array_backend(backend)
    calls = []
    gaussian_filter = B.gaussian_filter
    monkeypatch.setattr(B, "gaussian_filter", lambda *a, **k: calls.append(1) or gaussian_filter(*a, **k))
    rng = np.random.default_rng(0)
    vol = rng.random((16, 16, 16)).astype("float32")
    blocks = rng.random((2, 2, 2, 8, 8, 8)).astype("float32")
    xcorr = rng.random((2, 2, 2, 9, 9)).astype("float32")
    out = [
        warpfield.register.RegFilter()(vol),
        warpfield.register.Projector()(blocks, axis=-1),
        warpfield.register.Smoother()(xcorr, np.array([8, 8, 8])),
    ]
    assert all(isinstance(o, np.ndarray) for o in out)
    assert len(calls) >= 3


@pytest.mark.skipif(len(BACKENDS) < 2, reason="needs a GPU backend besides NumPy")
def test_warpmap_follows_backend_switch():
    """A WarpMap made before warpfield.set_backend() computes on the new backend afterwards, and so do the maps
    derived from it (one backend per process, also for existing maps)."""
    gpu = BACKENDS[0]
    fixed, moving = _small_volumes()
    recipe = warpfield.Recipe()
    recipe.levels = recipe.levels[:1]
    try:
        warpfield.set_backend(gpu)
        registered, wm, _ = warpfield.register_volumes(fixed, moving, recipe, verbose=False)
        assert backend_of(wm.warp_field).name == gpu
        warpfield.set_backend("numpy")
        assert wm.B.name == "numpy" and isinstance(wm.warp_field, np.ndarray)
        np.testing.assert_allclose(wm.warp(moving), registered, atol=1e-4)
        for derived in [wm.invert_fast(), wm.chain(wm), wm.median_filter()]:
            assert derived.B.name == "numpy" and isinstance(derived.warp_field, np.ndarray)
    finally:
        warpfield.set_backend("auto")


@pytest.mark.skipif(len(BACKENDS) < 2, reason="needs a GPU backend besides NumPy")
def test_registration_pyramid_after_backend_switch():
    """A RegistrationPyramid holds the reference on its backend: after a switch, using it gives a clear error."""
    fixed, moving = _small_volumes(size=32)
    recipe = warpfield.Recipe()
    recipe.levels = recipe.levels[:1]
    try:
        warpfield.set_backend(BACKENDS[0])
        pyramid = warpfield.register.RegistrationPyramid(fixed, recipe)
        warpfield.set_backend("numpy")
        with pytest.raises(RuntimeError, match="Create a new RegistrationPyramid"):
            pyramid.register_single(moving)
    finally:
        warpfield.set_backend("auto")


# --------------------------------------------------------------------------------------------- CuPy specifics
@requires_cupy
def test_registration_cupy_matches_numpy():
    """The full pipeline gives the same warp field on CuPy and on the NumPy backend."""
    fixed, moving = _small_volumes()
    recipe = warpfield.Recipe()
    recipe.add_level(block_size=[32, 32, 32], block_stride=0.5, repeats=2)
    fields = {}
    try:
        for name in ["cupy", "numpy"]:
            warpfield.set_backend(name)
            _, wm, _ = warpfield.register_volumes(fixed, moving, recipe, verbose=False)
            fields[name] = _to_numpy(wm.warp_field)
    finally:
        warpfield.set_backend("auto")
    np.testing.assert_allclose(fields["cupy"], fields["numpy"], atol=1e-3)


@requires_cupy
def test_cupy_warp_into_strided_out():
    """A non-contiguous `out` is filled correctly (the CUDA kernel writes a flat C-ordered array)."""
    import cupy as cp

    vol = cp.random.default_rng(0).random((20, 21, 22), dtype=cp.float32)
    wm = warpfield.WarpMap(np.zeros((3, 2, 2, 2)), [10, 11, 11], [10, 11, 11], vol.shape, vol.shape)
    buf = cp.zeros((20, 21, 44), dtype=cp.float32)
    out = buf[:, :, ::2]
    result = wm.warp(vol, out=out)
    assert result is out
    cp.testing.assert_allclose(out, vol, atol=1e-6)
    assert float(cp.abs(buf[:, :, 1::2]).max()) == 0  # nothing written between the strided elements


class _CudaArray:
    """Stands in for a non-CuPy CUDA array (PyTorch CUDA tensor, Numba device array, ...)."""

    def __init__(self, a):
        self._a, self.shape, self.dtype, self.ndim = a, a.shape, a.dtype, a.ndim

    @property
    def __cuda_array_interface__(self):
        return self._a.__cuda_array_interface__


@requires_cupy
def test_cuda_array_interface_inputs_stay_on_gpu():
    """Other CUDA arrays are used without a host copy and give CuPy results (as before 0.3)."""
    import cupy as cp
    from warpfield.ndimage import dogfilter

    vol = cp.random.default_rng(0).random((40, 40, 40), dtype=cp.float32)
    x = _CudaArray(vol)
    assert backend_of(x).name == "cupy"
    wm = warpfield.WarpMap(np.zeros((3, 2, 2, 2)), [20] * 3, [20] * 3, vol.shape, vol.shape)
    assert isinstance(wm.warp(x), cp.ndarray)
    assert isinstance(dogfilter(x), cp.ndarray)
    np.testing.assert_allclose(backend_of(x).to_numpy(x), vol.get())
    # and the NumPy backend can still take them
    expected = _to_numpy(dogfilter(vol))
    try:
        warpfield.set_backend("numpy")
        np.testing.assert_allclose(_to_numpy(dogfilter(x)), expected, atol=1e-4)
    finally:
        warpfield.set_backend("auto")


# ------------------------------------------------------------------------------------------------------- CLI
def test_cli_output_path_checks(tmp_path):
    data = str(tmp_path / "data.h5")
    # default: next to the moving file (the file part of an `.h5:` or `.h5/` path)
    assert _check_output_path(None, data, data) == str(tmp_path / "data_registered.h5")
    # an input file would be overwritten
    with pytest.raises(ValueError, match="input file"):
        _check_output_path(data, data, str(tmp_path / "other.h5"))
    with pytest.raises(ValueError, match="does not exist"):
        _check_output_path(str(tmp_path / "missing" / "out.h5"), data, data)


@pytest.mark.parametrize("sep", [":", "/"])
def test_cli_with_h5_dataset_paths(tmp_path, sep):
    """`file.h5:dataset` and `file.h5/dataset` inputs, without --output, write next to the input file."""
    fixed, moving = _small_volumes(size=48)
    data = tmp_path / "data.h5"
    with h5py.File(data, "w") as f:
        f["fixed"], f["moving"] = fixed, moving
    recipe = warpfield.Recipe()
    recipe.levels = recipe.levels[:1]
    recipe.to_yaml(tmp_path / "recipe.yml")
    cmd = [sys.executable, "-m", "warpfield", "--recipe", str(tmp_path / "recipe.yml"), "--backend", "numpy"]
    result = subprocess.run(
        cmd + ["--fixed", f"{data}{sep}fixed", "--moving", f"{data}{sep}moving"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    with h5py.File(tmp_path / "data_registered.h5", "r") as f:
        assert {"moving_reg", "recipe_json", "warp_map"} <= set(f)
    # refusing to overwrite the input leaves it intact
    result = subprocess.run(
        cmd + ["--fixed", f"{data}:fixed", "--moving", f"{data}:moving", "--output", str(data)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "input file" in result.stderr
    with h5py.File(data, "r") as f:
        assert set(f) == {"fixed", "moving"}


def test_cli_with_zarr_array_path(tmp_path):
    """`store.zarr/array` inputs, without --output, write next to the store (not into it)."""
    zarr = pytest.importorskip("zarr")
    fixed, moving = _small_volumes(size=48)
    group = zarr.open_group(str(tmp_path / "data.zarr"), mode="w")
    group.create_array("fixed", data=fixed)
    group.create_array("moving", data=moving)
    assert warpfield.load_data(str(tmp_path / "data.zarr" / "moving"))[1]["path"] == str(tmp_path / "data.zarr")
    recipe = warpfield.Recipe()
    recipe.levels = recipe.levels[:1]
    recipe.to_yaml(tmp_path / "recipe.yml")
    cmd = [sys.executable, "-m", "warpfield", "--recipe", str(tmp_path / "recipe.yml"), "--backend", "numpy"]
    cmd += ["--fixed", str(tmp_path / "data.zarr" / "fixed"), "--moving", str(tmp_path / "data.zarr" / "moving")]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    with h5py.File(tmp_path / "data_registered.h5", "r") as f:
        assert "moving_reg" in f
    assert not (tmp_path / "data.zarr" / "moving_registered.h5").exists()


def test_load_matlab_variable(tmp_path):
    """`file.mat:variable` loads the variable and reports the file as the path."""
    import scipy.io

    vol = np.arange(60, dtype="float32").reshape(3, 4, 5)
    scipy.io.savemat(tmp_path / "v.mat", {"vol": vol})
    data, info = warpfield.load_data(f"{tmp_path / 'v.mat'}:vol")
    np.testing.assert_array_equal(data, vol)
    assert info["path"] == str(tmp_path / "v.mat") and info["key"] == "vol"
    with pytest.raises(ValueError, match="not found"):
        warpfield.load_data(f"{tmp_path / 'v.mat'}:missing")


def _write_mat_v73(path, **arrays):
    """A minimal MATLAB v7.3 file: HDF5 with a 512-byte user block that starts with the MAT-file header."""
    with h5py.File(path, "w", userblock_size=512) as f:
        for name, a in arrays.items():
            f[name] = a.T  # MATLAB writes column-major data, which HDF5 sees with the axes reversed
    header = b"MATLAB 7.3 MAT-file, Platform: GLNXA64, Created on: Fri Nov 20 00:00:00 2026 HDF5 schema 1.00 ."
    with open(path, "r+b") as f:
        f.write(header.ljust(116) + b"\0" * 8 + b"\x00\x02IM")  # version 0x0200, little-endian marker "IM"


def test_load_matlab_v73_variable(tmp_path):
    """v7.3 files (HDF5 inside) load with the axes as in MATLAB, like older versions do through loadmat."""
    import scipy.io

    vol = np.arange(60, dtype="float32").reshape(3, 4, 5)
    _write_mat_v73(tmp_path / "v73.mat", vol=vol)
    with pytest.raises(NotImplementedError):  # what loadmat says about v7.3 files
        scipy.io.loadmat(tmp_path / "v73.mat")
    data, info = warpfield.load_data(f"{tmp_path / 'v73.mat'}:vol")
    np.testing.assert_array_equal(data, vol)
    assert info["path"] == str(tmp_path / "v73.mat") and info["key"] == "vol"
    with pytest.raises(ValueError, match="not found"):
        warpfield.load_data(f"{tmp_path / 'v73.mat'}:missing")


@pytest.mark.parametrize("fmt", ["h5", "mat", "mat73", "zarr"])
@pytest.mark.parametrize("sep", [":", "/", "\\"])
def test_load_data_name_after_file(tmp_path, fmt, sep):
    """HDF5, MATLAB and Zarr all take the dataset/variable/array after the file name, with `:` or `/` (or `\\`, as
    in Windows paths)."""
    import scipy.io

    vol = np.arange(60, dtype="float32").reshape(3, 4, 5)
    if fmt == "h5":
        path, name = tmp_path / "d.h5", "grp/vol"
        with h5py.File(path, "w") as f:
            f[name] = vol
    elif fmt == "mat":
        path, name = tmp_path / "d.mat", "vol"
        scipy.io.savemat(path, {name: vol})
    elif fmt == "mat73":
        path, name = tmp_path / "d.mat", "vol"
        _write_mat_v73(path, vol=vol)
    else:
        zarr = pytest.importorskip("zarr")
        path, name = tmp_path / "d.zarr", "grp/vol"
        zarr.open_group(str(path), mode="w").create_array(name, data=vol)
    given = name.replace("/", sep) if sep == "\\" else name  # Windows style: backslashes inside the name too
    data, info = warpfield.load_data(f"{path}{sep}{given}")
    np.testing.assert_array_equal(data, vol)
    assert info["path"] == str(path) and info["key"] == name


def test_load_data_container_paths(tmp_path):
    """No name where one is needed gives a clear error; a Zarr array store needs no name; folders whose names end
    in .h5 do not confuse the split."""
    with h5py.File(tmp_path / "d.h5", "w") as f:
        f["vol"] = np.ones((2, 3, 4))
    for path in [tmp_path / "d.h5", f"{tmp_path / 'd.h5'}/"]:
        with pytest.raises(ValueError, match="Add the dataset"):
            warpfield.load_data(path)
    (tmp_path / "exp.h5").mkdir()
    with h5py.File(tmp_path / "exp.h5" / "x.h5", "w") as f:
        f["vol"] = np.zeros((2, 3, 4))
    data, info = warpfield.load_data(f"{tmp_path / 'exp.h5' / 'x.h5'}:vol")
    assert info["path"] == str(tmp_path / "exp.h5" / "x.h5") and data.shape == (2, 3, 4)
    zarr = pytest.importorskip("zarr")
    zarr.create_array(str(tmp_path / "a.zarr"), data=np.ones((2, 3, 4), "float32"))
    data, info = warpfield.load_data(str(tmp_path / "a.zarr"))
    assert data.shape == (2, 3, 4) and info["path"] == str(tmp_path / "a.zarr") and info["key"] is None


def test_cli_with_different_shapes(tmp_path):
    """Fixed and moving volumes of different shapes register; --invert gives the fixed volume in moving space."""
    import scipy.ndimage

    rng = np.random.default_rng(0)
    big = scipy.ndimage.gaussian_filter(rng.random((64, 72, 80)).astype("float32"), 2)
    fixed, moving = big[4:60, 4:68, 4:76], np.roll(big, 2, axis=0)  # fixed (56, 64, 72), moving (64, 72, 80)
    np.save(tmp_path / "fixed.npy", fixed)
    np.save(tmp_path / "moving.npy", moving)
    recipe = warpfield.Recipe()
    recipe.levels = recipe.levels[:1]
    recipe.to_yaml(tmp_path / "recipe.yml")
    cmd = [sys.executable, "-m", "warpfield", "--recipe", str(tmp_path / "recipe.yml"), "--backend", "numpy"]
    cmd += ["--fixed", str(tmp_path / "fixed.npy"), "--moving", str(tmp_path / "moving.npy"), "--invert"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    with h5py.File(tmp_path / "moving_registered.h5", "r") as f:
        assert f["moving_reg"].shape == fixed.shape
        inner = (slice(6, -6),) * 3
        err = np.mean((f["moving_reg"][:][inner] - fixed[inner]) ** 2)
        err_centre_crop = np.mean((moving[4:60, 4:68, 4:76][inner] - fixed[inner]) ** 2)
        assert err < 0.05 * err_centre_crop
        assert f["fixed_reg_inv"].shape == moving.shape
        assert tuple(f["warp_map/ref_shape"][:]) == fixed.shape and tuple(f["warp_map/mov_shape"][:]) == moving.shape


def test_load_data_group_instead_of_array(tmp_path):
    """A group (HDF5 group, MATLAB v7.3 struct, Zarr group store) given instead of an array says what it holds,
    instead of loading the member names as data."""
    with h5py.File(tmp_path / "d.h5", "w") as f:
        f["grp/vol"] = np.ones((2, 3, 4))
    with pytest.raises(ValueError, match="is a group.*vol"):
        warpfield.load_data(f"{tmp_path / 'd.h5'}:grp")
    _write_mat_v73(tmp_path / "s.mat")
    with h5py.File(tmp_path / "s.mat", "a") as f:
        f["s/field"] = np.ones((2, 3, 4))
    with pytest.raises(ValueError, match="is a group.*field"):
        warpfield.load_data(f"{tmp_path / 's.mat'}:s")
    zarr = pytest.importorskip("zarr")
    zarr.open_group(str(tmp_path / "g.zarr"), mode="w").create_array("vol", data=np.ones((2, 3, 4), "float32"))
    with pytest.raises(ValueError, match="is a group.*vol"):
        warpfield.load_data(str(tmp_path / "g.zarr"))


def test_load_data_other_format_in_folder_named_like_hdf5(tmp_path):
    """A folder whose name ends in .h5 (or .mat, .zarr-like names) does not hide files of other formats inside it."""
    import tifffile

    vol = np.arange(24, dtype="float32").reshape(2, 3, 4)
    (tmp_path / "dir.h5").mkdir()
    tifffile.imwrite(tmp_path / "dir.h5" / "v.tif", vol)
    np.testing.assert_array_equal(warpfield.load_data(str(tmp_path / "dir.h5" / "v.tif"))[0], vol)
    (tmp_path / "dir.mat").mkdir()
    np.save(tmp_path / "dir.mat" / "v.npy", vol)
    tifffile.imwrite(tmp_path / "dir.mat" / "v.tif", vol)
    np.testing.assert_array_equal(warpfield.load_data(str(tmp_path / "dir.mat" / "v.tif"))[0], vol)


@pytest.mark.skipif("mlx" in BACKENDS, reason="MLX is a GPU backend that CUDA_VISIBLE_DEVICES does not hide")
def test_cli_without_gpu_fails_before_loading(tmp_path):
    """Without a GPU, the CLI stops before loading the volumes, and its hint is the CLI option."""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    missing = str(tmp_path / "not_there.npy")  # loading it would fail differently
    cmd = [sys.executable, "-m", "warpfield", "--fixed", missing, "--moving", missing, "--recipe", "default.yml"]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert result.returncode != 0
    assert "No GPU backend found" in result.stderr and "--backend numpy" in result.stderr, result.stderr
    assert "not_there" not in result.stderr and "Loading" not in result.stderr


# ------------------------------------------------------------------------------------------------------ docs
def test_submodules_are_documented():
    """pdoc only builds pages for names in __all__; the README links to the submodule pages."""
    for name in ["register", "ndimage", "warp", "utils", "backends"]:
        assert name in warpfield.__all__ and hasattr(warpfield, name)


# ------------------------------------------------------------------------------------------- installation
def test_no_gpu_backend_error_explains_install(monkeypatch):
    """Without a usable GPU backend, the error names the pip extras and why an installed CuPy is unusable."""
    from warpfield.backends import selection

    monkeypatch.setitem(selection._USABLE, "cupy", lambda: False)
    monkeypatch.setitem(selection._USABLE, "mlx", lambda: False)
    monkeypatch.setattr(selection, "_default", None)
    monkeypatch.setitem(selection._unusable_reason, "cupy", "ImportError: libcublasLt.so.12 not found")
    with pytest.raises(RuntimeError, match=r"warpfield\[cuda12\]") as err:
        selection.get_array_backend("auto")
    assert "libcublasLt" in str(err.value)
