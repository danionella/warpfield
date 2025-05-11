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
    fixed = np.random.rand(256, 256, 256).astype("float32")
    moving = np.roll(fixed, shift=5, axis=0).copy()  # Simulate a simple shift
    recipe = warpfield.Recipe.from_yaml("default.yml")

    registered, warp_map, _ = warpfield.register_volumes(fixed, moving, recipe, verbose=False)

    assert registered.shape == fixed.shape, "Registered volume shape mismatch."
    assert (
        np.abs(registered[10:-10, 10:-10, 10:-10] - fixed[10:-10, 10:-10, 10:-10]) < 0.2
    ).mean() > 0.9, "Registered volume does not match the fixed volume."
    assert warp_map is not None, "WarpMap object was not returned."


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
