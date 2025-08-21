# Copilot Instructions for warpfield

## Repository Overview

**warpfield** is a GPU-accelerated 3D non-rigid volumetric image registration library written in Python. It provides both a Python API and command-line interface for registering medical/scientific volumetric image data using GPU-based algorithms.

### Key Repository Information
- **Project Type**: Scientific Python library with CLI
- **Primary Language**: Python 3.9+
- **Key Dependencies**: CuPy (CUDA), NumPy, SciPy, PyDantic, H5Py
- **Target Platform**: Linux (recommended), Windows with CUDA support
- **Repository Size**: ~10 core source files, medium complexity
- **License**: MIT

### Critical Hardware Requirements
- **CUDA-compatible GPU required** - The library cannot function without GPU acceleration
- **GPU Memory**: ≥30 bytes per voxel (30 GB/gigavoxel) for processing volumes
- All tests and core functionality require GPU; they will skip/fail on CPU-only systems

## Build and Validation Instructions

### Environment Setup (ALWAYS Required)
The project **requires** CUDA and CuPy. Always use conda for initial setup:

```bash
# Recommended: Create conda environment from environment.yml
conda env create -n warpfield -f environment.yml
conda activate warpfield

# Alternative: Manual conda setup
conda create -n warpfield python>=3.9 cupy cuda-nvcc -c conda-forge
conda activate warpfield
```

### Installation and Build
1. **Install package in development mode**:
   ```bash
   pip install -e .
   ```
   - This installs the package with all dependencies
   - Takes 2-5 minutes depending on network
   - **Always run this before making changes**

2. **Install testing dependencies**:
   ```bash
   pip install pytest
   ```

### Testing (GPU Required)
```bash
# Run all tests (requires GPU)
pytest tests/

# Tests will SKIP if no GPU detected with warnings:
# "No GPU detected. Skipping GPU tests."
```

**Test Behavior**: 
- All meaningful tests require GPU and will skip on CPU-only systems
- Test execution time: ~30 seconds with GPU
- Tests create temporary volumes and verify registration accuracy

### CLI Validation
```bash
# Test CLI help
python -m warpfield --help

# Full CLI test requires sample data:
python -m warpfield --fixed fixed.npy --moving moving.npy --recipe default.yml --output result.h5
```

### Common Build Issues and Solutions

1. **ImportError: libcublas.so.12**: CUDA libraries not installed
   - **Solution**: Use conda to install CUDA: `conda install cupy cuda-nvcc -c conda-forge`

2. **"No GPU detected" warnings**: Running on CPU-only system
   - **Solution**: Tests will skip but package installs successfully

3. **Memory errors during tests**: Insufficient GPU memory
   - **Solution**: Tests use 256³ voxel volumes requiring ~4GB GPU RAM

## Project Layout and Architecture

### Source Code Structure
```
src/warpfield/
├── __init__.py          # Main imports: register_volumes, Recipe, load_data
├── __main__.py          # CLI entry point with argparse
├── register.py          # Core algorithm: WarpMap, WarpMapper, register_volumes()
├── utils.py             # Data loading: load_data() for .npy, .h5, .nii, .tiff
├── warp.py              # Volume warping operations
├── ndimage.py           # GPU-accelerated image processing functions
└── recipes/             # YAML configuration files
    ├── default.yml      # Standard registration recipe
    └── default_opm.yml  # Optimized recipe variant
```

### Key Classes and Functions
- **`register_volumes(fixed, moving, recipe)`**: Main registration function
- **`Recipe.from_yaml(path)`**: Load registration parameters from YAML
- **`WarpMap`**: Represents computed transformation between volumes
- **`load_data(path)`**: Universal data loader for scientific formats

### Configuration Files
- **`pyproject.toml`**: Modern Python packaging, entry points, pytest config
- **`environment.yml`**: Conda dependencies including CUDA requirements
- **`default.yml`**: Default registration recipe with 4 levels (translation → affine → 2 non-rigid)

### GitHub Workflows and CI/CD
Located in `.github/workflows/`:
- **`test.yml`**: Runs pytest on Ubuntu/Windows with micromamba + cupy setup
- **`publish_pypi.yml`**: PyPI publication (manual trigger)
- **`create_pdoc.yaml`**: Documentation generation with pdoc

**CI Build Process**:
1. Setup micromamba with cupy
2. `pip install pytest` 
3. `pip install -e .`
4. `pytest tests/`

### Supported Data Formats
The `load_data()` function supports:
- **NumPy**: `.npy` files
- **HDF5**: `.h5`, `.hdf5` with dataset specification (e.g., `file.h5:dataset`)
- **NIfTI**: `.nii`, `.nii.gz` medical imaging format
- **TIFF**: `.tiff`, `.tif` multi-page volumes
- **NRRD**: `.nrrd` scientific format
- **Zarr**: `.zarr` chunked arrays

### Recipe System
Registration recipes are YAML files defining multi-level registration:
- **Translation level**: Global alignment (`block_size: [-1, -1, -1]`)
- **Affine level**: Linear transformations (`affine: true`)
- **Non-rigid levels**: Local deformation (`block_size: [64,64,64]`, `[32,32,32]`)

Example recipe structure:
```yaml
pre_filter:
  clip_thresh: 0
  dog: true
levels:
  - block_size: [-1, -1, -1]  # Translation
    repeats: 1
  - block_size: [64, 64, 64]  # Non-rigid
    smooth:
      sigmas: [2.0, 2.0, 2.0]
    repeats: 5
```

## Development Workflow

### Making Changes
1. **Always activate conda environment first**
2. **Install in development mode**: `pip install -e .`
3. **Run tests after changes**: `pytest tests/` (if GPU available)
4. **Test CLI functionality** with sample data
5. **Check imports**: `python -c "import warpfield; print('OK')"`

### Code Style
- Uses Black formatter (line-length: 120) as configured in `pyproject.toml`
- No explicit linting configuration - relies on Black for consistency
- Docstrings follow Google style for pdoc documentation generation

### Common Development Tasks
- **Add new registration level**: Modify recipe YAML files
- **Support new data format**: Extend `load_data()` in `utils.py`
- **Modify algorithms**: Work in `register.py` (core) or `ndimage.py` (GPU ops)
- **Add CLI options**: Edit `__main__.py` argparse configuration

## Important Notes for Agents

1. **Trust these instructions** - GPU requirements and conda setup are non-negotiable
2. **Always check GPU availability** before running tests or core functionality  
3. **Use conda for environment setup** - pip alone will not work reliably
4. **Test both Python API and CLI** when making changes to core functionality
5. **Sample data creation**: Tests generate synthetic 256³ volumes with simple transformations
6. **Memory considerations**: Large volumes require significant GPU memory; adjust test sizes accordingly

The codebase is well-structured but has strict GPU requirements. Focus on understanding the Recipe system and WarpMap architecture when implementing registration-related changes.