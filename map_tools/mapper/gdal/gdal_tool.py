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

import os.path
import subprocess


def gdal_create_subpic(
    x: int, y: int, xsize: int, ysize: int, inname: str, outname: str
):
    """Creates a subpicture of a given picture, starting at the given x,y, with the given sizes."""
    print(
        f"Executing GDAL translate: {x}, {y}, {xsize}, {ysize}, {inname}, {outname}"
    )
    result = subprocess.run(
        [
            "gdal_translate",
            "-srcwin",
            f"{x}",
            f"{y}",
            f"{xsize}",
            f"{ysize}",
            inname,
            outname,
        ]
    )
    print(result)


def gdal_convert_to_format(image_path: str, output_format: str = "png") -> str:
    """
    Converts the image to the given format.
    :return: the path of the converted image, or empty string if not converted.
    """
    image_name_without_ext, ext = os.path.splitext(image_path)
    if ext == output_format:
        print(
            f"Not converting: image {image_path} already has extension {output_format}"
        )
        return ""

    converted_image_path = f"{image_name_without_ext}.{output_format}"
    print(
        f"Executing GDAL translate to convert to {output_format}: {image_path}, {converted_image_path}"
    )

    # GDAL_PAM_ENABLED = NO avoids creating intermediary xml files.
    result = subprocess.run(
        [
            "gdal_translate",
            "-of",
            output_format.upper(),
            image_path,
            converted_image_path,
            "--config",
            "GDAL_PAM_ENABLED",
            "NO",
        ]
    )
    print(result)

    return converted_image_path
