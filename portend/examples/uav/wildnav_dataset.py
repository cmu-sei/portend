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

import csv
import os
from pathlib import Path
from typing import Any

from portend.datasets.image_dataset import ImageDataSet
from portend.utils.typing import SequenceLike

COORDINATES_KEY = "coordinates"
PHOTO_CSV_FILE = "photo_metadata.csv"


class WildnavDataSet(ImageDataSet):
    # Overriden. Generates input file for this dataset.
    def post_process(self, dataset_config: dict[str, Any]):
        super().post_process(dataset_config)
        self._generate_csv_input()

    # Overriden (implemented abstract method).
    def get_model_inputs(self) -> list[SequenceLike]:
        """Returns the image inputs, but also adds a CSV file with additional data."""
        # Since the model inputs are the image files, we add this CSV as just another file that is passed to the algorithm, but as a separate input.
        model_inputs = super().get_model_inputs()
        model_inputs.append([self._csv_input_file_name()])
        return model_inputs

    def _csv_input_file_name(self) -> str:
        """Returns the name of the CSV input file that Wildnav expects."""
        return os.path.join(self.image_folder, PHOTO_CSV_FILE)

    def _generate_csv_input(self):
        """Generate photo csv file neded for input in Wildnav."""
        csv_input_file = self._csv_input_file_name()
        print(f"Generating CSV input file at: {csv_input_file}")
        header = [
            "Filename",
            "Latitude",
            "Longitude",
            "Altitude",
            "Gimball_Roll",
            "Gimball_Yaw",
            "Gimball_Pitch",
            "Flight_Roll",
            "Flight_Yaw",
            "Flight_Pitch",
        ]
        with open(csv_input_file, "w", encoding="UTF8") as f:
            writer = csv.writer(f)
            writer.writerow(header)

            # Write one line for each image with the coordinates, plus zero for the additional flight data we don't have.
            for image_id in self.get_ids():
                sample = self.get_sample_by_id(image_id)
                image_file = Path(sample[self.image_path_key]).name
                coordinates: list[float] = sample[COORDINATES_KEY]
                lat = coordinates[0]
                long = coordinates[1]
                line = [image_file, lat, long, 0, 0, 0, 0, 0, 0, 0]
                writer.writerow(line)
