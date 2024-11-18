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

from typing import Any, Optional

from portend.datasets.image_dataset import ImageDataSet
from portend.utils import exif_gps


class GPSImageDataSet(ImageDataSet):
    """A dataset for handling a list of images with GPS data in their EXIF metadata"""

    DEFAULT_COORDINATES_KEY = "coordinates"

    # Overriden.
    @classmethod
    def create_dataset_from_folder(
        cls,
        image_folder: Optional[str],
        extensions: Optional[list[str]] = [],
        fields: Optional[dict[str, Any]] = {},
        existing_data: Optional[str] = "",
    ):
        """Creates a JSON file with the expected ImageDataSet format for all images in a given folder."""
        print(f"Creating GPSImageDataset dataset from folder {image_folder}")
        image_dataset: ImageDataSet = super().create_dataset_from_folder(
            image_folder, extensions, fields, existing_data
        )

        # Go over each sample, load their EXIF GPS data, and update the sample with the coordinates.
        new_samples = []
        for sample in image_dataset.get_samples():
            path = sample[cls.DEFAULT_IMAGE_PATH_KEY]
            coordinates = exif_gps.get_exif_gps_decimal_coordinates(path)

            # Default to 0,0
            if len(coordinates) == 0:
                coordinates = [0, 0]

            sample.update({cls.DEFAULT_COORDINATES_KEY: coordinates})
            new_samples.append(sample)

        image_dataset.set_samples(new_samples)
        return image_dataset
