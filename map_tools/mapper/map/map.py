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

import os
import os.path
from pathlib import Path
from typing import Any, Optional

from mapper.gdal import gdal_tool
from mapper.image import image_split
from mapper.io import wildnav_csv_generator

DEFAULT_SUB_SIZE = 1000
DEFAULT_OUTPUT_FORMAT = "png"
MAP_DATA_FILE = "map.csv"


def create_map_data_file(
    output_folder: str, parts_gps_info: list[dict[str, Any]]
):
    """Creates a CSV file with data about the parts of a map, from the provided base image and GPS info coordinates."""
    if len(parts_gps_info) > 0:
        print("Creating CSV with coordinates from each subimage.")
        map_csv_filename = os.path.join(output_folder, MAP_DATA_FILE)
        wildnav_csv_generator.write_map_csv_coordinates(
            parts_gps_info, map_csv_filename
        )


def split_map(
    image_path: str, size: Optional[int] = None
) -> list[dict[str, Any]]:
    """
    Splits a map into subimages of the given format.
    :return: A list with dicts with info about the output images.
    """
    if size is None:
        size = DEFAULT_SUB_SIZE

    print(f"Starting image splitting for {image_path} to size {size}")
    out_images = image_split.split_image(image_path, size)
    return out_images


def convert_images(
    image_paths: list[str],
    images_info: list[dict[str, Any]],
    out_format: Optional[str] = None,
    remove_original: Optional[bool] = True,
) -> list[dict[str, Any]]:
    """
    Converts a list of images to the given format.
    :returns: update image info.
    """
    if out_format is None:
        out_format = DEFAULT_OUTPUT_FORMAT

    print(f"Converting image parts to {image_paths}, if needed")
    for image_path in image_paths:
        converted_image_path = gdal_tool.gdal_convert_to_format(
            image_path, out_format
        )
        if converted_image_path != "" and remove_original:
            print(f"Removing initial image {image_path}")
            os.remove(image_path)

            # Update image info with converted image name.
            for image_info in images_info:
                if image_info["filename"] == Path(image_path).name:
                    image_info["filename"] = Path(converted_image_path).name
                    break

    return images_info
