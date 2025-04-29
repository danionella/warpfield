# Run: pytest tests/test_all.py
# All functions to be tested should start with test_prefix

import os
import warnings
import subprocess

import pytest
import numpy as np
import h5py
import nibabel as nib
import pydicom
import tifffile

from warpfield.utils import import_data
from warpfield.register import register_volumes
from warpfield.recipes import from_yaml

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

    loaded_data, meta = import_data(str(file_path))
    assert np.allclose(loaded_data, data), "Loaded .npy data does not match expected data."


def test_import_h5(tmp_path):
    """Test importing a 3D .h5 file."""
    file_path = tmp_path / "test.h5"
    data = np.random.rand(10, 10, 10).astype("float32")
    with h5py.File(file_path, "w") as f:
        f.create_dataset("dataset", data=data)

    loaded_data, meta = import_data(f"{file_path}:dataset")
    assert np.allclose(loaded_data, data), "Loaded .h5 data does not match expected data."


def test_import_nii(tmp_path):
    """Test importing a 3D .nii file."""
    file_path = tmp_path / "test.nii"
    data = np.random.rand(10, 10, 10).astype("float32")
    nii = nib.Nifti1Image(data, affine=np.eye(4))
    nib.save(nii, file_path)

    loaded_data, meta = import_data(str(file_path))
    assert np.allclose(loaded_data, data), "Loaded .nii data does not match expected data."


def test_import_dicom(tmp_path):
    """Test importing a 3D DICOM file."""
    file_path = os.path.join(tmp_path, "test.dcm")
    data = np.random.randint(0, 256, (10, 10, 10), dtype="uint8")  # 3D data

    # Create the File Meta Information
    file_meta = pydicom.Dataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    file_meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID

    # Create the main DICOM dataset
    dicom = pydicom.Dataset()
    dicom.file_meta = file_meta
    dicom.PixelData = data.tobytes()
    dicom.Rows, dicom.Columns = data.shape[1], data.shape[2]
    dicom.NumberOfFrames = data.shape[0]
    dicom.BitsAllocated = 8  # Number of bits for each pixel
    dicom.BitsStored = 8
    dicom.HighBit = 7
    dicom.SamplesPerPixel = 1
    dicom.PixelRepresentation = 0  # 0 = unsigned, 1 = signed
    dicom.PhotometricInterpretation = "MONOCHROME2"
    dicom.Modality = "OT"  # Other
    dicom.StudyInstanceUID = pydicom.uid.generate_uid()
    dicom.SeriesInstanceUID = pydicom.uid.generate_uid()
    dicom.SOPInstanceUID = pydicom.uid.generate_uid()
    dicom.SOPClassUID = pydicom.uid.SecondaryCaptureImageStorage

    # Save the DICOM file with the correct structure
    pydicom.filewriter.dcmwrite(file_path, dicom, enforce_file_format=True)

    # Test the import_data function
    loaded_data, meta = import_data(str(file_path))
    assert loaded_data.shape == data.shape, "Loaded .dcm data shape does not match expected data."
    assert np.array_equal(loaded_data, data), "Loaded .dcm data does not match expected data."


def test_import_tiff(tmp_path):
    """Test importing a 3D .tiff file."""
    file_path = tmp_path / "test.tiff"
    data = np.random.rand(10, 10, 10).astype("float32")
    tifffile.imwrite(file_path, data)

    loaded_data, meta = import_data(str(file_path))
    assert np.allclose(loaded_data, data), "Loaded .tiff data does not match expected data."


@pytest.mark.skipif(not gpu_available, reason="No GPU detected.")
def test_register_volumes():
    """Test the register_volumes function."""
    fixed = np.random.rand(256, 256, 256).astype("float32")
    moving = np.roll(fixed, shift=5, axis=0).copy()  # Simulate a simple shift
    recipe = from_yaml("default.yml")

    registered, warp_map, _ = register_volumes(fixed, moving, recipe, verbose=False)

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
