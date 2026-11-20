import argparse
import os
import logging
import numpy as np
import h5py
import hdf5plugin

from .utils import load_data
from .register import register_volumes, Recipe

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def _check_output_path(output, fixed_file, moving_file):
    """Return the output path (default: next to the moving file) and check it before the registration runs.

    ``fixed_file`` and ``moving_file`` are the file parts of the input paths (without an HDF5 dataset key).
    """
    output_path = output or f"{os.path.splitext(moving_file)[0]}_registered.h5"
    if os.path.realpath(output_path) in (os.path.realpath(fixed_file), os.path.realpath(moving_file)):
        raise ValueError(f"Output path {output_path} is an input file; it would be overwritten. Pass a different --output.")
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if not os.path.isdir(output_dir):
        raise ValueError(f"Output directory {output_dir} does not exist.")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="GPU-accelerated volumetric image registration tool.")
    parser.add_argument(
        "--fixed",
        required=True,
        help="Path to the fixed volume. For HDF5, MATLAB and Zarr files, add the dataset after the file name: "
        "data.h5:name or data.h5/name.",
    )
    parser.add_argument("--moving", required=True, help="Path to the moving volume (same formats as --fixed).")
    parser.add_argument(
        "--recipe",
        required=True,
        help="Path to a recipe YAML file, or a bundled recipe (default.yml, default_opm.yml).",
    )
    parser.add_argument("--output", help="Path to save the registered image/volume.")
    parser.add_argument(
        "--compression",
        type=str,
        default="gzip",
        help="Compression method for saving the registered volume (default: gzip).",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Also invert the warp map and warp the fixed image into the space of the moving image (saved as fixed_reg_inv).",
    )
    parser.add_argument(
        "--backend", default="auto", help="Backend to compute on: `cupy`, `mlx` or `numpy` (CPU, slow). Default: auto (first available GPU backend)."
    )
    args = parser.parse_args()

    # load
    logging.info(f"Loading fixed image from {args.fixed}...")
    fixed_image, fixed_info = load_data(args.fixed)
    logging.info(f"Loading moving image from {args.moving}...")
    moving_image, moving_info = load_data(args.moving)
    output_path = _check_output_path(args.output, fixed_info["path"], moving_info["path"])
    if fixed_image.ndim != 3 or moving_image.ndim != 3:
        raise ValueError("Both fixed and moving images must be 3D volumes.")
    if fixed_image.shape != moving_image.shape:
        raise ValueError("Fixed and moving images must have the same shape.")
    # report shape
    logging.info(f"Volume shapes: {fixed_image.shape}")
    recipe = Recipe.from_yaml(args.recipe)

    # register
    logging.info("Registering the moving image to the fixed image...")
    registered_image, warp_map, _ = register_volumes(fixed_image, moving_image, recipe, backend=args.backend, verbose=True)

    if args.invert:
        logging.info("Inverting the warp map...")
        warp_map_inv = warp_map.invert_fast()
        logging.info("Registering the fixed image to the moving image...")
        fixed_reg_inv = warp_map_inv.apply(moving_image)

    # save
    # consider args.invert when logging and reporting fixed_reg_inv:
    logging.info(
        f"Saving to {output_path}:/moving_reg, {'...:/fixed_reg_inv, ' if args.invert else ''}and :/warp_map..."
    )
    with h5py.File(output_path, "w") as f:
        f.create_dataset("moving_reg", data=registered_image, compression=args.compression)
        f.create_dataset("recipe_json", data=recipe.model_dump_json().encode("utf-8"))
        if args.invert:
            f.create_dataset("fixed_reg_inv", data=fixed_reg_inv, compression=args.compression)
    warp_map.to_h5(output_path, group="warp_map", compression=args.compression)

    logging.info("Done!")


if __name__ == "__main__":
    main()
