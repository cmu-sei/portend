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
import typing
from typing import Any, Callable, Tuple

import mapper.gps.exif_gps as exif_gps
import mapper.image.image_merge as image_merge
import requests
from mapper.io import file_utils
from mapper.map import map
from mapper.tiles.tile_system import TileSystem

# from PIL import Image

TILE_SOURCES = {
    "ARCGIS": {
        "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{}/{}/{}",
        "format": lambda url, x, y, z: url.format(z, y, x),
    },
    "Google": {
        "url": "https://mt1.google.com/vt/lyrs=s&x={}&y={}&z={}",
        "format": lambda url, x, y, z: url.format(x, y, z),
    },
    "Bing": {
        "url": "http://ecn.t1.tiles.virtualearth.net/tiles/a{}.png?g=441&mkt=en-us&n=z",
        "format": lambda url, x, y, z: url.format(
            TileSystem.tile_xy_to_quad_key(x, y, z)
        ),
    },
}

IMAGE_TYPES = {"image/jpeg": "jpg", "image/png": "png"}
COMBINED_SUFFIX = "_combined"


def download_tiles(
    zoom_level: int,
    lat: float,
    long: float,
    source: str = "ARCGIS",
    output_folder: str = "./",
    radius: int = 0,
) -> Tuple[list[list[str]], list[dict[str, Any]]]:
    """Saves into a file the tile for the given zoom, lat and log, using the provided tile source and output folder."""
    # Check valid source.
    if source not in TILE_SOURCES:
        raise Exception(
            f"Source {source} is unknown. Known sources: {TILE_SOURCES.keys()}"
        )

    # First obtain tile coordinates.
    center_tile_x, center_tile_y = TileSystem.lat_long_to_tile_xy(
        lat, long, zoom_level
    )
    print(f"Tile: ({center_tile_x}, {center_tile_y})")

    # Delete and create folder as needed.
    print(f"Deleting old folder tile data from {output_folder}")
    file_utils.delete_folder_data(output_folder)
    file_utils.create_folder(output_folder)

    # Iterate over all tiles we need.
    base_url: str = typing.cast(str, TILE_SOURCES[source]["url"])
    formatting: Callable[[str, int, int, int], str] = typing.cast(
        Callable[[str, int, int, int], str], TILE_SOURCES[source]["format"]
    )
    image_data: list[dict[str, Any]] = []
    all_images: list[list[str]] = []
    curr_row = -1
    for tile_y in range(center_tile_y - radius, center_tile_y + radius + 1):
        all_images.append([])
        curr_row += 1
        for tile_x in range(center_tile_x - radius, center_tile_x + radius + 1):
            # Construct proper URL.
            tile_url: str = formatting(base_url, tile_x, tile_y, zoom_level)

            # Get tile.
            print(f"Downloading tile from url: {tile_url}")
            response = requests.get(tile_url)
            if response:
                # Check image type.
                image_type = response.headers.get("content-type")
                print(f"Got image with format {image_type}")
                if image_type not in IMAGE_TYPES:
                    raise Exception(f"Image type not supported: {image_type}")

                # Prepare file name and folder to save image to.
                file_name = f"tile_z_{zoom_level}_tile_{tile_x}_{tile_y}.{IMAGE_TYPES[image_type]}"
                file_path = os.path.join(output_folder, file_name)

                print(f"Writing output file to: {file_path}")
                with open(file_path, "wb") as f:
                    f.write(response.content)
                all_images[curr_row].append(file_path)

                # Calculate coordinates of center of this tile.
                center_lat, center_long = TileSystem.tile_xy_to_lat_long_center(
                    tile_x, tile_y, zoom_level
                )
                image_data.append(
                    {
                        "filename": file_name,
                        "path": file_path,
                        "center_lat": center_lat,
                        "center_long": center_long,
                        "tile_x": tile_x,
                        "tile_y": tile_y,
                    }
                )

                # Add coordinates as exif data.
                exif_gps.add_geolocation(file_path, center_lat, center_long)
            else:
                print(f"Error: {response.reason}")

    return all_images, image_data


def _calculate_map_gps(
    image_name: str,
    image_path: str,
    images_data: list[dict[str, Any]],
    zoom_level: int,
    tile_x: int,
    tile_y: int,
) -> dict[str, Any]:
    """Calculate GPS coordinates information for center and corners of the provided map."""
    # Calculate coordinates of center of this tile.
    center_lat, center_long = TileSystem.tile_xy_to_lat_long_center(
        tile_x, tile_y, zoom_level
    )
    print(f"CLA, CLO: {center_lat}, {center_long}")

    if len(images_data) == 0:
        raise Exception("Error: images data must have at least one tile data.")

    # For the top left, get info about the first tile/image.
    top_left_tile = images_data[0]
    top_left_lat, top_left_long = TileSystem.tile_xy_to_lat_long_top_left(
        top_left_tile["tile_x"], top_left_tile["tile_y"], zoom_level
    )
    print(f"TLA, TLO: {top_left_lat}, {top_left_long}")

    # For bottom right, get info about the last time/image.
    bottom_right_tile = images_data[len(images_data) - 1]
    (
        bottom_right_lat,
        bottom_right_long,
    ) = TileSystem.tile_xy_to_lat_long_bottom_right(
        bottom_right_tile["tile_x"], bottom_right_tile["tile_y"], zoom_level
    )
    print(f"BRLA, BRLO: {bottom_right_lat}, {bottom_right_long}")

    # Put everything together in a dict.
    gps_info = {}
    gps_info["filename"] = image_name
    gps_info["file_path"] = image_path
    gps_info["center_lat"] = center_lat
    gps_info["center_long"] = center_long
    gps_info["top_left_lat"] = top_left_lat
    gps_info["top_left_long"] = top_left_long
    gps_info["bottom_right_lat"] = bottom_right_lat
    gps_info["bottom_right_long"] = bottom_right_long

    return gps_info


def _calculate_images_gps(
    images_info: list[dict[str, Any]],
    main_image_info: dict[str, Any],
    zoom_level: int,
) -> list[dict[str, Any]]:
    """
    Calculates the GPS coordinates for each image, based on the main one.
    :return: a list of GPS dicts for each sub image.
    """
    all_info = []

    # Reference pixel from main image.
    (
        main_top_left_x_pixel,
        main_top_left_y_pixel,
    ) = TileSystem.lat_long_to_pixel_xy(
        main_image_info["top_left_lat"],
        main_image_info["top_left_long"],
        zoom_level,
    )

    for image_info in images_info:
        # The lat and long for the top left can be obtained through the pixels, adding the offset to the base one.
        top_left_x_pixel = main_top_left_x_pixel + image_info["x_offset"]
        top_left_y_pixel = main_top_left_y_pixel + image_info["y_offset"]
        top_left_lat, top_left_long = TileSystem.pixel_xy_to_lat_long(
            top_left_x_pixel, top_left_y_pixel, zoom_level
        )

        # The lat and long for the bottom right can be obtained through the pixels, adding the offset and the size to the base one.
        bottom_left_x_pixel = top_left_x_pixel + image_info["x_size"]
        bottom_left_y_pixel = top_left_y_pixel + image_info["y_size"]
        bottom_right_lat, bottom_right_long = TileSystem.pixel_xy_to_lat_long(
            bottom_left_x_pixel, bottom_left_y_pixel, zoom_level
        )

        # The lat and long for the center can be obtained through the pixels, adding the offset and half the size.
        center_x_pixel = top_left_x_pixel + image_info["x_size"] // 2
        center_y_pixel = top_left_y_pixel + image_info["y_size"] // 2
        center_lat, center_long = TileSystem.pixel_xy_to_lat_long(
            center_x_pixel, center_y_pixel, zoom_level
        )

        image_info["center_lat"] = center_lat
        image_info["center_long"] = center_long
        image_info["top_left_lat"] = top_left_lat
        image_info["top_left_long"] = top_left_long
        image_info["bottom_right_lat"] = bottom_right_lat
        image_info["bottom_right_long"] = bottom_right_long

        all_info.append(image_info)

    return all_info


def _create_map_from_tiles(
    all_images: list[list[Any]],
    images_data: list[dict[str, Any]],
    zoom_level: int,
    lat: float,
    long: float,
    output_folder: str = "./",
    radius: int = 0,
) -> dict[str, Any]:
    """
    Creates a combined map from a set of images.
    :return: A dictionary with the map path, and GPS info about it.
    """
    center_tile_x, center_tile_y = TileSystem.lat_long_to_tile_xy(
        lat, long, zoom_level
    )
    combined_image_name = f"tile_z_{zoom_level}_tile_{center_tile_x}_{center_tile_y}_r_{radius}{COMBINED_SUFFIX}.{IMAGE_TYPES['image/png']}"
    combined_image_path = os.path.join(output_folder, combined_image_name)
    print(f"Combining images into output file: {combined_image_path}")
    image_merge.combine_images(all_images, combined_image_path)

    # Calculate coordinates of this combined image.
    map_info = _calculate_map_gps(
        combined_image_name,
        combined_image_path,
        images_data,
        zoom_level,
        center_tile_x,
        center_tile_y,
    )

    # Add coordinates as exif data.
    exif_gps.add_geolocation(
        combined_image_path, map_info["center_lat"], map_info["center_long"]
    )

    return map_info


def create_maps(
    all_images: list[list[str]],
    images_data: list[dict[str, Any]],
    zoom_level: int,
    lat: float,
    long: float,
    output_folder: str = "./",
    radius: int = 0,
    remove_tiles: bool = False,
    max_size: int = map.DEFAULT_SUB_SIZE,
) -> list[dict[str, Any]]:
    """
    Saves into a file the tile for the given zoom, lat and log, using the provided tile source and output folder.
    :return: a list of dictionary with information about the map or its parts.
    """
    # Combine images.
    all_maps_info = []
    if len(all_images) > 1:
        # Delete and create folder as needed.
        print(f"Deleting old folder map data from {output_folder}")
        file_utils.delete_folder_data(output_folder)
        file_utils.create_folder(output_folder)

        # Create merged map and get its GPS info.
        map_info = _create_map_from_tiles(
            all_images,
            images_data,
            zoom_level,
            lat,
            long,
            output_folder,
            radius,
        )
        all_maps_info.append(map_info)

        if remove_tiles:
            print("Removing individual tiles")
            for image_row in all_images:
                for image in image_row:
                    os.remove(image)

        # If width or height of resulting combined map is more than our max size, split.
        # combined_image = Image.open(map_info["file_path"])
        # width, height = combined_image.size
        # if max_size != 0 and (width > max_size or height > max_size):

        # Split map by size and calculate GPS data for each part.
        sub_images_data = map.split_map(map_info["file_path"], max_size)
        all_maps_info = _calculate_images_gps(
            sub_images_data, map_info, zoom_level
        )

    return all_maps_info
