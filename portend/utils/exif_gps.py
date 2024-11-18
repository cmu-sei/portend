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

from typing import Any

from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS
from PIL.Image import Exif

GPS_KEY_NAME = "GPSInfo"


def get_exif(file_name: str) -> Exif:
    image: Image.Image = Image.open(file_name)
    return image.getexif()


def get_exif_with_gps(file_name: str) -> dict[Any, Any]:
    exif_data: Exif = get_exif(file_name)
    exif_gps = {}
    if exif_data is not None:
        gps_ifd = 0
        for key, value in TAGS.items():
            if value == GPS_KEY_NAME:
                gps_ifd = key
                break

        if gps_ifd != 0:
            gps_info = exif_data.get_ifd(gps_ifd)
            exif_gps = {
                GPSTAGS.get(key, key): value for key, value in gps_info.items()
            }

    return exif_gps


def get_exif_gps_decimal_coordinates(filename: str) -> list[float]:
    """Returns the EXIF GPS coordinates embedded in the given file in a list as decimal [lat, long]."""
    info = get_exif_with_gps(filename)
    if len(info) == 0:
        return []

    gps_dict = {}
    for key in ["Latitude", "Longitude"]:
        if "GPS" + key in info and "GPS" + key + "Ref" in info:
            e = info["GPS" + key]
            ref = info["GPS" + key + "Ref"]
            gps_dict[key] = (
                float(e[0]) + float(e[1]) / 60 + float(e[2]) / 3600
            ) * (-1 if ref in ["S", "W"] else 1)

    gps_list = []
    if "Latitude" in gps_dict and "Longitude" in gps_dict:
        gps_list = [gps_dict["Latitude"], gps_dict["Longitude"]]

    return gps_list
