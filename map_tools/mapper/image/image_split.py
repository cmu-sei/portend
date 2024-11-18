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

import os.path
from pathlib import Path
from typing import Any

import cv2
from mapper.gdal import gdal_tool

COMBINED_SUFFIX = "_combined"


def _structure_image_info(
    x_offset: int, y_offset: int, x_size: int, y_size: int, image_path: str
) -> dict[str, Any]:
    """
    Calculates the GPS coordinates for the given image.
    :return: a dict with GPS info for the image.
    """
    image_info: dict[str, Any] = {}
    image_info["filename"] = Path(image_path).name
    image_info["file_path"] = image_path
    image_info["x_offset"] = x_offset
    image_info["y_offset"] = y_offset
    image_info["x_size"] = x_size
    image_info["y_size"] = y_size

    return image_info


def split_image(image_path: str, line_size: int) -> list[dict[str, Any]]:
    """
    Creates subpictures for a given image, preserving GeoTIFF data if any.
    :param: line_size: the max amount of pixels to have per dimension (both width and height) for each part.
    :return: a list of dicts with information about the split images.
    """
    out_images: list[dict[str, Any]] = []
    print(f"Creating subimages of max size {line_size}x{line_size}")

    # Open image and get dimensions.
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise Exception(f"Image {image_path} was not found.")
    ysize = img.shape[0]
    xsize = img.shape[1]
    print(f"Original map file is {xsize} x {ysize}")

    # Divide image in sub images of SUB_SIZE x SUB_SIZE.
    num_x_cuts = xsize // line_size
    num_y_cuts = ysize // line_size

    # If the image size is not dividable by SUB_SIZE, we will need another image of a smaller size.
    if num_x_cuts * line_size < xsize:
        num_x_cuts += 1
    if num_y_cuts * line_size < ysize:
        num_y_cuts += 1
    print(f"Number of x cuts: {num_x_cuts}, number of y cuts: {num_y_cuts}")

    # Create all images, looping over a mesh.
    x_offset = 0
    y_offset = 0
    for i in range(0, num_x_cuts):
        # Update the offset, and if we are at the end of a row, update the final size.
        x_offset = i * line_size
        curr_x_size = line_size
        if xsize - x_offset < line_size:
            curr_x_size = xsize - x_offset

        # Same with y.
        for j in range(0, num_y_cuts):
            y_offset = j * line_size
            curr_y_size = line_size
            if ysize - y_offset < line_size:
                curr_y_size = ysize - y_offset

            # Give a name to the new subimage.
            image_path_without_ext, ext = os.path.splitext(image_path)
            image_path_without_ext = image_path_without_ext.replace(
                COMBINED_SUFFIX, ""
            )
            output_image_path = f"{image_path_without_ext}_{i}_{j}{ext}"

            # Create each sub image.
            gdal_tool.gdal_create_subpic(
                x_offset,
                y_offset,
                curr_x_size,
                curr_y_size,
                image_path,
                output_image_path,
            )
            image_info = _structure_image_info(
                x_offset, y_offset, curr_x_size, curr_y_size, output_image_path
            )
            out_images.append(image_info)

    return out_images
