# Run: pytest tests/test_all.py
# All functions to be tested should start with test_prefix

import os
import warnings
import subprocess

import pytest
import numpy as np
import h5py
import nibabel as nib
import tifffile

import warpfield
from warpfield.utils import load_data

try:
    import cupy as cp

    _ = cp.cuda.runtime.getDeviceCount()  # Check if any GPU devices are available
    gpu_available = True
except (ImportError, cp.cuda.runtime.CUDARuntimeError):
    gpu_available = False
    warnings.warn("No GPU detected. Skipping GPU tests.")


def test_trivial():
    assert True == True


def test_trivial2():
    assert False == False


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


@pytest.mark.skipif(not gpu_available, reason="No GPU detected.")
def test_register_volumes():
    """Test the register_volumes function."""
    import cupyx.scipy.ndimage

    fixed = np.random.rand(256, 256, 256).astype("float32")
    fixed = cupyx.scipy.ndimage.gaussian_filter(cp.array(fixed), sigma=1).get()
    moving = np.roll(fixed, shift=5, axis=0).copy()  # Simulate a simple shift
    recipe = warpfield.Recipe.from_yaml("default.yml")

    registered, warp_map, _ = warpfield.register_volumes(fixed, moving, recipe, verbose=False)

    assert registered.shape == fixed.shape, "Registered volume shape mismatch."
    assert (
        np.abs(registered[10:-10, 10:-10, 10:-10] - fixed[10:-10, 10:-10, 10:-10]) < 0.2
    ).mean() > 0.9, "Registered volume does not match the fixed volume."
    assert warp_map is not None, "WarpMap object was not returned."


@pytest.mark.skipif(not gpu_available, reason="No GPU detected.")
def test_push_pull_coordinates_roundtrip():
    """Pulling pushed coordinates should return the original coordinates."""
    ref_shape = (128, 128, 128)
    block_size = np.array([16, 16, 16])
    grid = np.indices((8, 8, 8)).astype("float32")
    # smooth displacement field with shifts of a few voxels
    warp_field = np.stack([3.0 + np.sin(grid[1] / 3), -2.0 + np.cos(grid[2] / 4), 2.5 * np.ones_like(grid[0])]).astype(
        "float32"
    )
    warp_map = warpfield.register.WarpMap(
        warp_field, block_size, block_stride=block_size, ref_shape=ref_shape, mov_shape=ref_shape
    )

    points = np.random.default_rng(0).uniform(32, 96, size=(3, 200)).astype("float32")
    pushed = warp_map.push_coordinates(points)
    pulled = warp_map.pull_coordinates(pushed).get()

    assert np.abs(pulled - points).max() < 0.25, "pull_coordinates does not reverse push_coordinates."


@pytest.mark.skipif(not gpu_available, reason="No GPU detected.")
def test_smoother_shear():
    """Smoother with shear should run, and shear=0 should match the non-shear path."""
    xcorr_proj = cp.array(np.random.default_rng(0).standard_normal((6, 6, 6, 9, 9)).astype("float32"))
    block_size = np.array([32, 32, 32])
    sigmas = [1.0, 1.5, 0.8]

    plain = warpfield.register.Smoother(sigmas=sigmas)(xcorr_proj.copy(), block_size)
    sheared0 = warpfield.register.Smoother(sigmas=sigmas, shear=0.0)(xcorr_proj.copy(), block_size)
    sheared = warpfield.register.Smoother(sigmas=sigmas, shear=0.5)(xcorr_proj.copy(), block_size)

    assert sheared.shape == xcorr_proj.shape
    assert np.allclose(sheared0.get(), plain.get(), atol=1e-5), "shear=0 should match the non-shear smoothing."
    assert not np.allclose(sheared.get(), plain.get(), atol=1e-3), "shear=0.5 should change the result."


@pytest.mark.skipif(not gpu_available, reason="No GPU detected.")
def test_warp_volume_single_point_field():
    """A displacement field with one point per axis (translation level) is a constant shift (#45)."""
    import scipy.ndimage
    from warpfield.warp import warp_volume

    vol = np.random.default_rng(0).standard_normal((20, 22, 24)).astype("float32")
    disp = np.float32([2.5, -1.0, 0.75]).reshape(3, 1, 1, 1)
    scale, offset = cp.asarray([20, 22, 24], dtype="float32"), cp.asarray([-0.5, -0.5, -0.5], dtype="float32")
    out = warp_volume(cp.asarray(vol), cp.asarray(disp), scale, offset).get()
    # the kernel interpolates towards 0 beyond the last voxel, as scipy's "grid-constant" does
    expected = scipy.ndimage.map_coordinates(vol, np.indices(vol.shape) + disp, order=1, mode="grid-constant", cval=0)
    np.testing.assert_allclose(out, expected, atol=1e-5)


@pytest.mark.skipif(not gpu_available, reason="No GPU detected.")
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
