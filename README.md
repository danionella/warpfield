![Python Version](https://img.shields.io/badge/python-3.9+-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![tests](https://github.com/danionella/warpfield/actions/workflows/test.yml/badge.svg)
[![PyPI - Version](https://img.shields.io/pypi/v/warpfield)](https://pypi.org/project/warpfield/)
[![Conda Version](https://img.shields.io/conda/v/conda-forge/warpfield)](https://anaconda.org/conda-forge/warpfield)
![GitHub last commit](https://img.shields.io/github/last-commit/danionella/warpfield)

<img width="1000" src="https://github.com/user-attachments/assets/09b2109d-5db6-4f7e-8e3d-000361455a0f"/> <br>
<sup>Example shown: region-level 3D registration across subjects (video slowed down ~5x). Cell-level registration of microscopy data is also possible.</sup>

# warpfield

A GPU-accelerated Python library for non-rigid 3D registration / warping.

Links: [API documentation](http://danionella.github.io/warpfield), [GitHub repository](https://github.com/danionella/warpfield)


### Features

- GPU-accelerated code for high performance ([CuPy](https://cupy.dev/), CUDA kernels & FFT plans)
- Typical speedup compared to CPU-based methods: > 1000x (**seconds vs. hours** for gigavoxel volumes)
- Forward and inverse transform of 3D volumes as well as point coordinates
- Python API and command-line interface (CLI)
- Support for `.h5`, `.nii`, `.tiff`, and `.npy` file formats

### General Principle

The registration process aligns a moving 3D volume to a fixed reference volume by estimating and applying a displacement field. The process is typically performed in a multi-resolution (pyramid) fashion, starting with coarse alignment and progressively refining the displacement field at finer resolutions.

The key steps are:

1. **Preprocessing**: Enhance features in the volumes (e.g., using Difference-of-Gaussian or 'DoG' filtering) to improve registration accuracy.
2. **Block Matching**: Divide the volumes into smaller 3D blocks, project them to 2D (along each axis) for memory and compute efficiency, and calculate 2D cross-correlation maps. After smoothing these 2D maps across neighboring blocks, use their maxima to determine the block-level 3D displacement vector.
3. **Displacement Field Estimation**: Combine block-level displacement vectors into a displacement field (and optionaly apply a median filter or fit an affine transform)
4. **Warping**: Apply the displacement field to the moving volume to align it with the fixed volume.
5. **Multi-Resolution Refinement**: Repeat the above steps at progressively finer resolutions to refine the alignment.

## Hardware requirements

- A computer running Linux (recommended) or Windows
- A CUDA-compatible GPU with sufficient GPU memory: ≥ 30 bytes per voxel (30 GB / gigavoxel) of your 3D volume


## Installation

Using [conda/mamba](https://github.com/conda-forge/miniforge?tab=readme-ov-file#install):

```bash
conda install conda-forge::warpfield
```

Or, to install into a new conda environment (recommended):

```bash
conda create -n warpfield conda-forge::warpfield
conda activate warpfield
```

Installation via pip is also possible, but not recommended (due to cuda dependencies):

```bash
pip install warpfield
```

For development, fork and clone the repository. Then type:

```bash
conda env create -n warpfield -f environment.yml
conda activate warpfield
pip install -e .
```

## Quickstart
```python
import warpfield 

# 1. Load data (note: the two volumes are expected to be of the same array shape and resolution)
vol_ref = np.load("reference_volume.npy")
vol_mov = np.load("moving_volume.npy")

# 2. Choose registration recipe (here: loaded from a YAML file. See below for alternatives)
recipe = warpfield.Recipe.from_yaml('default.yml')

# 3. Register moving volume
vol_mov_reg, warp_map, _ = warpfield.register_volume(vol_ref, vol_mov, recipe)

# 4. Optional: apply the transformation to another volume (same shape and resolution)
vol_another_reg = warp_map.apply(vol_another)

# 5. Optional: apply inverse transformation to the reference volume
vol_ref_reg = warp_map.invert_fast().apply(vol_ref)

# 6. Optional: apply the warp transformation to a set of coordiantes (3-by-n array, in units of volume voxels)
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

> [!NOTE]  
> Physical units, scalings or other metadata that may be present in data files are ignored. Fixed and moving volumes are expected to be of the same shape and resolution.

#### Required Arguments

- `--fixed`: Path to the fixed 3D volume file (`.h5`, `.nii`, `.tiff`, `.npy`). For hierarchical file formats (`.h5`), specify the dataset name using the format `filename.h5:dataset_name`. 
- `--moving`: Path to the moving 3D volume file.
- `--recipe`: Path to the registration recipe YAML file (`.yml`).

#### Optional Arguments

- `--output`: Path to save the registered volume (`.h5`). Defaults to `<moving>_registered.h5` if not provided.
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

The registration pipeline is defined by a recipe. The recipe consists of a pre-filter that is applied to all volumes (typically a DoG filter to sharpen features) and list of levels, each of which contains a set of parameters for the registration process. Typically, each level corresponds to a different resolution of the displacement field (the block size), with the first level being the coarsest and the last level being the finest.

### Recipe parameters

| Pre-filter parameter      | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| `clip_thresh`     | Pixel value threshold for clipping each volume. Default is 0. This setting helps remove DC background, which can otherwise cause edges (the shifted volume is 0 outside the FOV).  |
| `dog`             | If True, apply a 3D Difference-of-Gaussians (DoG) pre-filter to each volume. Default is True.                 |
| `low`             | The σ<sub>low</sub> value for the 3D DoG pre-filter. Default is 0.5           |
| `high`            | The σ<sub>high</sub> value for the 3D DoG pre-filter. Default is 10.0. Note: σ<sub>low</sub> and σ<sub>high</sub> should be smaller and bigger than the feature of interest, respectively. A σ of 1 correponds to a FWHM of ~ 2.4.)             |


| Level parameter      | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| `block_size`      | Shape of blocks, into which the volume is divided. Positive numbers indicate block shape in voxels (e.g. [32, 16, 32]), while negative numbers are interpreted as "divide volume shape by this number" (e.g. [-5, -5, -5] results in 5 blocks along each axis, when block stride is 1.0)|
| `block_stride`    | Determines whether blocks overlap. Either list of 3 integer values (block center distances, or strides, in voxels) or scalar float (fraction of block_size). Default is 1.0 (no block overlap). Set this to a smaller value (e.g. 0.5) for overlaping blocks and higher precision, but larger (e.g. 8x) memory footprint   |
| `project.max`     | If True, apply 3D -> 2D max projections to each volume block. If false, apply mean projections. Default is True           |
| `project.dog`     | If True, apply a DoG filter to each 2D projection. Default is True           |
| `project.low`     | The σ<sub>low</sub> value for the 2D DoG filter. Default is 0.5 voxels (pixels).                 |
| `project.high`    | The σ<sub>high</sub> value for the 2D DoG filter. Default is 10.0 voxels (pixels).               |
| `project.normalize`    | Whether to normalize the projections before calculating the cross-covariance (which would make it a cross-correlation). Defaults to False or 0.0. Values can range from 0.0 (False) to 1.0 (True). Values in between imply normalisation by `l2_norm**project.normalize`            |
| `project.periodic_smooth` | If True, apply periodic smoothing. Default is False. This can help reduce block-edge effects (affecting FFT-based correlation). |
| `tukey_ref`    | if not None, apply a Tukey window to the reference projections (alpha = tukey_ref). Default is 0.5. The Tukey window can reduce block-edge effects. |
| `smooth.sigmas`   | Sigmas for smoothing cross-correlations across blocks. Default is [1.0, 1.0, 1.0] blocks. |
| `smooth.shear`    | Shear parameter (specific to oblique plane wobble – ignore otherwise). Default is None.                      |
| `smooth.long_range_ratio` | Long range ratio for double gaussian kernel. Default is None. To deal with empty or low contrast regions, a second smooth with a larger (5x) sigma is applied to the cross-correlation maps and added. Typical values are between 0 (or None) and 0.1
| `median_filter`   | If True, apply median filter to the displacement field. Default is True                  |
| `affine`        | If True, fit affine transformation to the displacement field. Default is False. The affine fit ignores all edge voxels (to reduce edge effects) and therefore needs at least 4 blocks along each axis |
| `update_rate`          | Update rate for the displacement field. Default is 1.0. This value can be lowered to dampen oscillations, if needed (rarely).|
| `repeats`          | Number of iterations for this level. More repeats allow each block to deviate further from neighbors, despite smoothing. Typical values range from 1-10. Disable a level by setting repeats to 0.|


### Defining recipes

Recipes can be loaded from YAML files (either those shipped with this package, such as [default.yml](https://github.com/danionella/warpfield/blob/main/src/warpfield/recipes/default.yml), or your own): 

```python
recipe = warpfield.Recipe.from_yaml("default.yml")
# or your own recipe:
# recipe = warpfield.recipes.from_yaml("path/to/your/recipe.yaml")

# You can then modify recipe parameters programmatically, e.g.;
recipe.pre_filter.clip_thresh=10
# ... etc.
```

Alternatively, you can define a recipe from scratch using the `Recipe` class. For example:

```python
# create a basic recipe with one affine registration level and a default pre-filter:
recipe = warpfield.Recipe()
recipe.pre_filter.clip_thresh = 0 # clip DC background, if present

# affine level properties
recipe.levels[0].repeats = 5

# add non-rigid registration levels:
recipe.add_level(block_size=[128,128,128])
recipe.levels[-1].smooth.sigmas = [1.0,1.0,1.0]
recipe.levels[-1].repeats = 5

recipe.add_level(block_size=[64,64,64])
recipe.levels[-1].block_stride = 0.5
recipe.levels[-1].smooth.sigmas = [1.0,1.0,1.0]
recipe.levels[-1].repeats = 5

recipe.add_level(block_size=[32,32,32])
recipe.levels[-1].block_stride = 0.5
recipe.levels[-1].project.low = 1
recipe.levels[-1].project.high = 2
recipe.levels[-1].smooth.sigmas = [4.0,4.0,4.0]
recipe.levels[-1].smooth.long_range_ratio = 0.1
recipe.levels[-1].repeats = 5
```

> [!TIP]
> The speed of `warpfield` enables rapid iterative optimization of the registration process. Start with a simple recipe, such as the one above. Deactivate all levels, except for the first affine level, by setting their `repeats` to 0. Confirm that the affine registration converged (increasing repeats should not change the result) and move on to the second level. If voxels are anisotropic, adjust `block_size` to make blocks roughly isotropic in real space. Inspect results as you change the settings and repeats of the second level, then add more fine-grained levels if necessary. Adjust `project.low` and `project.high` to the relevant feature size if needed (which may get smaller in finer levels). If the moving volume warps too much, consider larger blocks / fewer levels. Otherwise, increase `smooth.sigmas`, reduce repeats, or reduce `block_stride` to 0.5 if you can afford the increase in memory footprint and compute time. You may also want to provide `register_volumes` with a callback function (see [`register_volumes`](https://danionella.github.io/warpfield/warpfield/register.html#register_volumes) and tip below) to observe each level and repeats of the registration process. It is very helpful for troubleshooting and for reducing compute time by adjusting the number of levels and repeats to the necessary minimum.

> [!TIP]
> Generating videos of the registration process (you will need to `conda install imageio imageio-ffmpeg scikit-image`):
> ```python
> video_path = "output.mp4" 
> units_per_voxel = [1,1,1] # voxel aspect ratio or physical dimensions (e.g. µm)
> callback = warpfield.utils.mips_callback(units_per_voxel=units_per_voxel)
> moving_reg, warpmap, _ = warpfield.register_volumes(fixed, moving, recipe, video_path=video_path, callback=callback)
> ```
> See also [`register_volumes`](https://danionella.github.io/warpfield/warpfield/register.html#register_volumes) documentation and [`notebooks/example.ipynb`](https://github.com/danionella/warpfield/blob/main/notebooks/example.ipynb).

## Feedback and contributions

We value your feedback and contributions to improve `warpfield`! Here's how you can help:

- **Bug reports**: If you encounter any issues or unexpected behavior, please [open a bug report](https://github.com/danionella/warpfield/issues/new?assignees=&labels=bug&template=bug_report.md&title=%5BBUG%5D).
- **Feature requests**: Have an idea for a new feature or improvement? Let us know by [submitting a feature request](https://github.com/danionella/warpfield/issues/new?assignees=&labels=enhancement&template=feature_request.md&title=%5BFEATURE%5D).
- **General discussion**: Start a conversation in our [Discussions](https://github.com/danionella/warpfield/discussions). Whether you have questions, ideas, or just want to share how `warpfield` has helped your work, we'd like to hear from you!
- **Questions**: If you have questions about using `warpfield`, post them in [Q&A](https://github.com/danionella/warpfield/discussions/categories/q-a).

#### Contributing code or documentation

Whether it's fixing a bug, adding a feature, or improving documentation, your help is appreciated. To contribute, fork the repository, make your changes in a feature branch, and submit a pull request (PR) for review. Please follow these minimum standards:

- Code Quality: Write clean, readable, and well-documented code.
- Testing: Include tests for new features or bug fixes.
- If your change doesn't concern an existing issue or discussion, please create one first to explain the motivation behind your change.

Thank you for your support!

## Citing our work
If you use `warpfield` in your research, please cite the paper that first described our registration approach:

Hoffmann, M., Henninger, J. et al. Blazed oblique plane microscopy reveals scale-invariant inference of brain-wide population activity. Nature Communications 14, 8019 (2023). [https://doi.org/10.1038/s41467-023-43741-x](https://doi.org/10.1038/s41467-023-43741-x)

```bibtex
@article{Hoffmann2023,
  title={Blazed oblique plane microscopy reveals scale-invariant inference of brain-wide population activity},
  author={Hoffmann, Maximilian and Henninger, Jorg and Veith, Johannes and Richter, Lars and Judkewitz, Benjamin},
  journal={Nature Communications},
  volume={14},
  number={1},
  pages={8019},
  year={2023},
  publisher={Nature Publishing Group},
  doi={10.1038/s41467-023-43741-x}
}
```

## See also

- [Advanced Normalization Tools in Python (ANTsPy)](https://github.com/ANTsX/ANTsPy)
- [Computational Morphometry Toolkit (CMTK)](https://www.nitrc.org/projects/cmtk)
- [Constrained Large Deformation Diffeomorphic Image Registration (CLAIRE)
](https://github.com/andreasmang/claire)
