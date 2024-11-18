#
# Portend Toolset
#
# Copyright 2024 Carnegie Mellon University.
#
# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.
#
# Licensed under a MIT (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.
#
# [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use and distribution.
#
# This Software includes and/or makes use of Third-Party Software each subject to its own license.
#
# DM24-1299
#

from __future__ import annotations

import types
from typing import Any

from portend.datasets.image_dataset import ImageDataSet
from portend.utils import module as module_utils
from portend.utils.logging import print_and_log

IMAGE_DRIFT_PACKAGE = "portend.drifts.image."
DEFAULT_IMAGE_FORMAT = "img-{}-{}-drifted.png"


def load_sub_module(
    submodule_name: str, package: str = IMAGE_DRIFT_PACKAGE
) -> types.ModuleType:
    """Loads the drift submodule."""
    return module_utils.load_module(submodule_name, package)


def generate_drifted_image_names(
    original_names: list[str], name_format: str = DEFAULT_IMAGE_FORMAT
) -> list[str]:
    """Generates names for the drifted images based on the original name and the format defined above."""
    image_names = [
        name_format.format(i, image_name)
        for i, image_name in enumerate(original_names)
    ]
    return image_names


# Interface method to be called by drifter.
def apply_drift(
    base_dataset: ImageDataSet, params: dict[str, Any]
) -> ImageDataSet:
    """Applies drift on a given dataset"""

    # Call image drift generation function, which returns a list of the drifted images.
    submodule_name = params.get("submodule")
    if submodule_name is None:
        raise Exception("No submodule name was provided.")
    print_and_log(f"Starting image drift, for submodule: {submodule_name}")
    drift_submodule = load_sub_module(submodule_name)
    print_and_log(f"Drifting {len(base_dataset.image_list)} images...")
    drifted_images = drift_submodule.drift_images(
        base_dataset.image_list, params
    )

    # Create a new dataset with these new images.
    drifted_image_names = generate_drifted_image_names(base_dataset.image_names)
    drifted_dataset = ImageDataSet()
    drifted_dataset = base_dataset.clone_image_dataset(drifted_dataset)
    drifted_dataset.set_image_folder(params.get("img_output_dir"))
    drifted_dataset.set_images_from_list(drifted_images, drifted_image_names)

    return drifted_dataset
