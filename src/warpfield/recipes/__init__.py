import os
import pathlib

import yaml

from ..register import Recipe


def from_yaml(yaml_path: str = "default") -> Recipe:
    """
    Get a recipe from YAML by filename.
    """
    this_file_dir = pathlib.Path(__file__).resolve().parent
    if os.path.isfile(yaml_path):
        yaml_path = yaml_path
    else:
        yaml_path = os.path.join(this_file_dir, yaml_path)

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    return Recipe.model_validate(data)
