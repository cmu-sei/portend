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

import argparse
import os

import mapper.io.wildnav_csv_generator as wildnav_csv_generator
import mapper.tiles.tile_handler as tile_handler
from mapper.map import map

DEFAULT_PHOTO_ZOOM = 19
DRONE_CSV_FILE = "photo_metadata.csv"
DRONE_JSON_FILE = "dataset.json"
DEFAULT_OUTPUT_FOLDER = "./temp_io"
PHOTOS_FOLDER = "photos"
MAP_FOLDER = "map"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-z",
        type=int,
        default=DEFAULT_PHOTO_ZOOM,
        help="Zoom level to use for map, starting from 0",
    )
    parser.add_argument(
        "--lat", type=float, help="Latitud in degrees", required=True
    )
    parser.add_argument(
        "--long", type=float, help="Longitude in degrees", required=True
    )
    parser.add_argument(
        "--source",
        type=str,
        default="ARCGIS",
        help="Data source for the tile: ARCGIS or GoogleSat",
    )
    parser.add_argument(
        "-o", type=str, default=DEFAULT_OUTPUT_FOLDER, help="Base output folder"
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=0,
        help="How many additional tiles to get in a radius around the center one. 1 means all tiles 1 tile away (i.e., 8), 2, 2 tiles away, etc",
    )
    parser.add_argument(
        "--map",
        action="store_true",
        help="If present, also generate a map merging all pictures.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=0,
        help="Max pixel width size for a map image, split into this size as needed.",
    )
    arguments, _ = parser.parse_known_args()

    output_folder = arguments.o
    if output_folder is None:
        output_folder = DEFAULT_OUTPUT_FOLDER

    # Get all tiles.
    photo_output_folder = os.path.join(output_folder, PHOTOS_FOLDER)
    image_matrix, image_data = tile_handler.download_tiles(
        arguments.z,
        arguments.lat,
        arguments.long,
        source=arguments.source,
        output_folder=photo_output_folder,
        radius=arguments.radius,
    )

    # Generate CSV  for separate tiles.
    wildnav_csv_generator.write_drone_csv_coordinates(
        image_data, os.path.join(photo_output_folder, DRONE_CSV_FILE)
    )

    if arguments.map is not None:
        # Merge tiles, and generate CSV for map.
        map_output_folder = os.path.join(output_folder, MAP_FOLDER)
        map_data = tile_handler.create_maps(
            image_matrix,
            image_data,
            arguments.z,
            arguments.lat,
            arguments.long,
            output_folder=map_output_folder,
            radius=arguments.radius,
            max_size=arguments.size,
        )
        map.create_map_data_file(map_output_folder, map_data)
