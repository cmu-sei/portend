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

from mapper.gps import geotiff
from mapper.map import map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("IMAGE_PATH")
    parser.add_argument("--size")
    parser.add_argument("--if")
    parser.add_argument("--of")
    args = parser.parse_args()

    image_path = args.IMAGE_PATH
    size = int(args.size) if args.size else None
    out_format = args.of

    # Split the map into sub images, and extract GPS data for each new piece.
    sub_images_data = map.split_map(image_path, size)
    sub_images_paths = [image["file_path"] for image in sub_images_data]
    sub_images_info = geotiff.get_geotiff_gps_info(sub_images_paths)

    # Convert the sub images if needed, and create CSV file with GPS info.
    updated_sub_images_info = map.convert_images(
        sub_images_paths, sub_images_info, out_format
    )
    map.create_map_data_file(
        os.path.dirname(image_path), updated_sub_images_info
    )


if __name__ == "__main__":
    main()
