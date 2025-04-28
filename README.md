![Python Version](https://img.shields.io/badge/python-3.9+-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![tests](https://github.com/danionella/warpfield/actions/workflows/test.yml/badge.svg)
[![PyPI - Version](https://img.shields.io/pypi/v/warpfield)](https://pypi.org/project/warpfield/)
[![Conda Version](https://img.shields.io/conda/v/danionella/warpfield)](https://anaconda.org/danionella/warpfield)
![GitHub last commit](https://img.shields.io/github/last-commit/danionella/warpfield)

# warpfield

A GPU-accelerated Python library for non-rigid volumetric image registration / warping.

Links: [API documentation](http://danionella.github.io/warpfield), [GitHub repository](https://github.com/danionella/warpfield)

---

## Features

- GPU-accelerated kernels (CuPy, CuPy RawKernels & FFT plans) for high performance  
- Non-rigid registration via block-wise rigid cross-correlation
- `WarpMap` class to represent, compose, invert, and apply displacement fields  
- Fast Difference-of-Gaussian (DoG) filtering and a variety of projection methods (max, mean, max_dog, etc.)  
- Support for .h5, .npy, .nii and .tiff file formats
- Python API and command-line interface (CLI)

---

## Hardware requirements

- A computer running Linux (recommended) or Windows
- A CUDA-compatible GPU with sufficient GPU memory: ≥ 40 bytes per voxel (40 GB / gigavoxel) of your 3D volume


## Installation

We recommend installing all dependencies via **conda** or **mamba**. Adjust cuda-version in `environment.yml` to your system, if needed.

```bash
# Change into the repository root directory, then type:
mamba env create -n warpfield -f environment.yml
mamba activate warpfield
pip install -e .
```

## Quickstart
```python
import warpfield 

# 1. Load data (note: the two volumes are expected to be of the same array shape and resolution)
vol_ref = np.load("reference_volume.npy")
vol_mov = np.load("moving_volume.npy")

# 2. Choose registration recipe (here: loaded from a YAML file. See below for alternatives)
recipe = warpfield.recipes.from_yaml('default.yml')

# 3. Register moving volume
vol_mov_reg, warp_map = warpfield.register.register_volume(vol_ref, vol_mov, recipe)

# 4. Optional: apply inverse transformation 
vol_ref_reg = warp_map.invert_fast().unwarp(vol_ref)

# 5. Optional: apply the warp transformation to another volume
vol_another_reg = warp_map.unwarp(vol_another)

# 6. Optional: apply the warp transformation to a set of points (3-by-n array)
points = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
points_pushed = warp_map.push_coordinates(points)
points_pulled = warp_map.pull_coordinates(points) # inverse transformation
```

## Command-Line Interface (CLI)

The `warpfield` library provides a command-line interface. This allows you to perform registration directly from the terminal without writing Python code.

#### Usage

```bash
python -m warpfield --fixed <fixed_image_path> --moving <moving_image_path> --recipe <recipe_path> [options]
# You can use the `--help` flag to see detailed instructions for the CLI:
python -m warpfield --help
```

#### Required Arguments

- `--fixed`: Path to the fixed image/volume file (e.g., `.nii`, `.h5`, `.npy`, etc.).
- `--moving`: Path to the moving image/volume file (e.g., `.nii`, `.h5`, `.npy`, etc.).
- `--recipe`: Path to the recipe YAML file for registration.

#### Optional Arguments

- `--output`: Path to save the registered image/volume. Defaults to `<moving>_registered.h5` if not provided.
- `--compression`: Compression method for saving the registered volume. Default is `gzip`.
- `--invert`: Additionally, register the moving image to the fixed image.

#### Output Structure

The output file is an HDF5 file containing the following datasets:
- `/moving_reg`: The registered moving image.
- `/warp_map`: A group containing the warp field and its metadata:
  - `/warp_field`: The displacement field.
  - `/block_size`: The block size (in voxels).
  - `/block_stride`: The block stride (in voxels).
- `/fixed_reg_inv` (optional): The fixed image registered to the moving image (if `--invert` is used).

## Recipes

The registration pipeline is defined by a recipe. The recipe consists of a pre-filter (`RegFilter`) that is applied to all volumes (typically a DoG filter to sharpen features) and list of level descriptors (`LevelConfig`), each of which contains a set of parameters for the registration process. Typically, each level corresponds to a different resolution of the input volume, with the first level being the coarsest and the last level being the finest.

### Recipe parameters

| Pre-filter parameter      | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| `clip_thresh`     | Threshold for clipping each volume. Default is 0                  |
| `dog`             | If True, apply a 3D DoG pre-filter to each volume. Default is True                 |
| `low`             | The lower sigma value for the DoG pre-filter. Default is 0.5                   |
| `high`            | The higher sigma value for the DoG pre-filter. Default is 10.0                 |


| Level parameter      | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| `block_size`      | Shape of blocks, whose rigid displacement is estimated. Positive numbers indicate block shape (e.g. [32, 16, 32]), whle negative numbers are interpreted as "divide axis into this many blocks" (e.g. [-5, -5, -5])|
| `block_stride`    | Stride. Either list of int stride sizes or float (fraction of block_size). Default is 1. Set to smaller value – e.g. 0.5 – for higher precision, but larger memory footprint   |
| `project.max`     | If True, apply a max filter to the volume block. Default is True           |
| `project.dog`     | If True, apply a DoG filter to the volume block. Default is True           |
| `project.low`     | The lower sigma value for the 2D DoG filter. Default is 0.5 voxels. (Note: A sigma of 1 correponds to a FWHM of ~ 2.4. The low and high sigmas should be smaller and bigger than the feature of interest. )                    |
| `project.high`    | The higher sigma value for the 2D DoG filter. Default is 10.0 voxels               |
| `smooth.sigmas`   | Sigmas for smoothing cross-correlations across blocks. Default is [1.0, 1.0, 1.0] blocks. |
| `smooth.truncate` | Truncate parameter for gaussian kernel. Default is 5 blocks.                      |
| `smooth.shear`    | Shear parameter (specific to oblique plane wobble). Default is None.                      |
| `smooth.long_range_ratio` | Long range ratio (0..1) for double gaussian kernel. Default is None. To deal with empty or low contrast regions, a second smooth with a larger (5x) sigma is applied to the cross-correlation maps and added. |
| `median_filter`   | If True, apply median filter to the displacement field. Default is True                  |
| `affinify`        | If True, apply affine transformation to the displacement field. Default is False          |
| `repeat`          | Number of iterations for this level. Default is 1         |


### Defining recipes

Recipes can be loaded from YAML files (either those shipped with the package or your own): 

```python
recipe = warpfield.recipes.from_yaml("default.yml")
# or your own recipe:
# recipe = warpfield.recipes.from_yaml("path/to/your/recipe.yaml")
```

You can then modify the recipe parameters as needed (this is a convenient option as long as the number of levels doesn't change). For example:

```python
recipe.pre_filter.clip_thresh=10

recipe.levels[0].block_size = [-5,-5,-5]
recipe.levels[0].smooth.sigmas=[1,1,1]
recipe.levels[0].affinify = True
recipe.levels[0].median_filter = False
recipe.levels[0].repeat = 10

recipe.levels[1].block_size = [-20,-10,-40]
recipe.levels[1].smooth.sigmas=[2,2,2]
recipe.levels[1].smooth.long_range_ratio = 0.05
recipe.levels[1].repeat = 5

recipe.levels[2].block_size = [32, 8, 32]
recipe.levels[2].block_stride = 0.5
recipe.levels[2].smooth.sigmas=[4,4,4]
recipe.levels[2].smooth.long_range_ratio = 0.1
recipe.levels[2].project.low = 1
recipe.levels[2].project.high = 2
recipe.levels[2].repeat = 5

print(f'recipe has {len(recipe.levels)} levels')
```


Alternatively, you can define a recipe from scratch using the `Recipe` class and its components. For example:

```python
from warpfield.register import Recipe, RegFilter, LevelConfig, Smoother, Projector, RegistrationPyramid

recipe = Recipe(
    pre_filter= RegFilter(clip_thresh=10),
    levels=[
        LevelConfig(block_size=[-5, -5, -5], 
                    smooth=Smoother(sigmas=[1, 1, 1]), 
                    project=Projector(low=2, high=10), 
                    affinify=True, 
                    median_filter=False,
                    repeat=20
                   ),
        LevelConfig(block_size=[-10, -10, -10], 
                    smooth=Smoother(sigmas=[2, 2, 2]), 
                    project=Projector(low=2, high=10), 
                    repeat=10),        
        LevelConfig(block_size=[32, 32, 32], 
                    block_stride=0.5,
                    smooth=Smoother(sigmas=[2, 2, 2]), 
                    project=Projector(low=0.5, high=2), 
                    repeat=5),
    ]
)
```

See [default.yml](./src/warpfield/recipes/default.yml) for an example recipe YML file.
