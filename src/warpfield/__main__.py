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


def main():
    parser = argparse.ArgumentParser(description="GPU-accelerated volumetric image registration tool.")
    parser.add_argument("--fixed", required=True, help="Path to the fixed image/volume file.")
    parser.add_argument("--moving", required=True, help="Path to the moving image/volume file.")
    parser.add_argument("--recipe", required=True, help="Path to the recipe YAML file for registration.")
    parser.add_argument("--output", help="Path to save the registered image/volume.")
    parser.add_argument(
        "--compression",
        type=str,
        default="gzip",
        help="Compression method for saving the registered volume (default: gzip).",
    )
    parser.add_argument(
        "--invert", action="store_true", help="Invert the warp map and register the moving image to the fixed image."
    )
    args = parser.parse_args()
    output_path = args.output or f"{os.path.splitext(args.moving)[0]}_registered.h5"

    # load
    logging.info(f"Loading fixed image from {args.fixed}...")
    fixed_image, _ = load_data(args.fixed)
    logging.info(f"Loading moving image from {args.moving}...")
    moving_image, _ = load_data(args.moving)
    if fixed_image.ndim != 3 or moving_image.ndim != 3:
        raise ValueError("Both fixed and moving images must be 3D volumes.")
    if fixed_image.shape != moving_image.shape:
        raise ValueError("Fixed and moving images must have the same shape.")
    # report shape
    logging.info(f"Volume shapes: {fixed_image.shape}")
    recipe = Recipe.from_yaml(args.recipe)

    # register
    logging.info("Registering the moving image to the fixed image...")
    registered_image, warp_map, _ = register_volumes(fixed_image, moving_image, recipe, verbose=True)

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
        warp_map_group = f.create_group("warp_map")
        warp_map_group.create_dataset("warp_field", data=warp_map.warp_field.get(), compression=args.compression)
        warp_map_group.create_dataset("block_size", data=warp_map.block_size.get())
        warp_map_group.create_dataset("block_stride", data=warp_map.block_stride.get())
        if args.invert:
            f.create_dataset("fixed_reg_inv", data=fixed_reg_inv, compression=args.compression)

    logging.info("Done!")


if __name__ == "__main__":
    main()
