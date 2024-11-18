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

import typing
from pathlib import Path
from typing import Any, Tuple

from osgeo import gdal, osr


def get_geotiff_gps_info(geotiff_images: list[str]) -> list[dict[str, Any]]:
    """
    Gest coordinate info from the provided GeoTIFF images.
    :return: A dictionary with entries for the coordinates of the top left and bottom right corners.
    """
    images_info: list[Any] = []
    try:
        # Now info for each image.
        for image_path in geotiff_images:
            # For each image, get the name and change ext to the final output.
            info: dict[str, Any] = {}
            info["filename"] = Path(image_path).name
            info["file_path"] = image_path
            geotiff_image = GeoTIFFImage(image_path)

            (
                info["top_left_long"],
                info["top_left_lat"],
                _,
            ) = geotiff_image.top_left_coords()

            (
                info["bottom_right_long"],
                info["bottom_right_lat"],
                _,
            ) = geotiff_image.bottom_right_coords()

            images_info.append(info)
    except RuntimeError as ex:
        print(
            f"Could not generate metadata from images, they may not be GeoTIFF files: {ex}"
        )

    return images_info


class GeoTIFFImage:
    """Converts coordinates from a GeoTIFF base image."""

    def __init__(self, image_path: str) -> None:
        self.gdal_image = gdal.Open(image_path)
        (
            self.c,
            self.a,
            self.b,
            self.f,
            self.d,
            self.e,
        ) = self.gdal_image.GetGeoTransform()

        projection_name = self.gdal_image.GetProjection()
        if projection_name == "":
            raise RuntimeError(
                "Can't initiate conversor, image does not have GeoTIFF projection info."
            )

        srs = osr.SpatialReference()
        srs.ImportFromWkt(projection_name)
        srsLatLong = srs.CloneGeogCS()
        self.coord_transform = osr.CoordinateTransformation(srs, srsLatLong)

    def pixel_to_coord(
        self, col: float, row: float
    ) -> Tuple[float, float, float]:
        """Returns global coordinates to pixel center using base-0 raster index"""
        xp = self.a * col + self.b * row + self.c
        yp = self.d * col + self.e * row + self.f
        coords = typing.cast(
            Tuple[float, float, float],
            self.coord_transform.TransformPoint(xp, yp, 0),
        )
        return coords

    def top_left_coords(self) -> Tuple[float, float, float]:
        return self.pixel_to_coord(0, 0)

    def bottom_right_coords(self) -> Tuple[float, float, float]:
        return self.pixel_to_coord(
            self.gdal_image.RasterXSize, self.gdal_image.RasterYSize
        )
