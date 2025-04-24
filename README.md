![Python Version](https://img.shields.io/badge/python-3.9+-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![tests](https://github.com/danionella/warpfield/actions/workflows/test.yml/badge.svg)
[![PyPI - Version](https://img.shields.io/pypi/v/warpfield)](https://pypi.org/project/warpfield/)
[![Conda Version](https://img.shields.io/conda/v/danionella/warpfield)](https://anaconda.org/danionella/warpfield)
![GitHub last commit](https://img.shields.io/github/last-commit/danionella/warpfield)

# warpfield

A GPU-accelerated Python library for block-wise volumetric image registration and warping.

Links: [API documentation](http://danionella.github.io/warpfield), [GitHub repository](https://github.com/danionella/warpfield)

---

## Features

- GPU-accelerated kernels (CuPy, CuPy RawKernels & FFT plans) for high performance  
- Piece-wise rigid registration via block-wise cross-correlation
- Fast Difference-of-Gaussian (DoG) filtering and a variety of projection methods (max, mean, max_dog, etc.)  
- `WarpMap` class to represent, compose, invert, and apply displacement fields  

---

## Installation

You can install warpfield via **conda** or **mamba**.


```bash
# Create & activate a new environment
mamba create -n warpfield -f environment.yml
mamba activate warpfield
# Change into repository root directory
pip install -e .
```

## Quickstart
```python
from warpfield.register import Recipe, RegFilter, LevelConfig, Smoother, Projector, RegistrationPyramid

# 1. Load data
vol_ref = np.load("reference_volume.npy")
vol_mov = np.load("moving_volume.npy")

# 2. Choose registration recipe
# set up recipe

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
        LevelConfig(block_size=[-20, -10, -40], 
                    smooth=Smoother(sigmas=[2, 2, 2]), 
                    project=Projector(low=2, high=10), 
                    repeat=10),
        LevelConfig(block_size=[32, 8, 32], 
                    block_stride=0.5,
                    smooth=Smoother(sigmas=[4, 4, 4], long_range_ratio=0.01), 
                    project=Projector(low=1, high=2),
                    repeat=3),
    ]
)

rp = RegistrationPyramid(vol_ref, recipe=recipe)

# 3. Register a new volume
registered_vol, warp_map, _ = rp.register_single(vol_mov)

# 4. Apply the combined warp to another volume
registered_vol_2 = warp_map.unwarp(vol_mov_2)

# 5. clear GPU memory
del rp
```

## Contributing

- Fork the repo and/or create a feature branch.
- Use Google style docstrings.
- Write tests for new functionality.
- Submit a pull request—CI will run linting & tests automatically.