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

import numpy as np

from portend.datasets import dataset
from portend.utils.typing import SequenceLike


class IcebergDataSet(dataset.DataSet):
    """A dataset following the Kaggle competition format of SAR data."""

    BAND1_KEY = "band_1"
    BAND2_KEY = "band_2"
    ANGLE_KEY = "inc_angle"

    BAND_WIDTH = 75
    BAND_HEIGHT = 75
    BAND_DEPTH = 3

    x_combined_bands = np.empty((0, BAND_WIDTH, BAND_HEIGHT, BAND_DEPTH))

    # Overriden.
    def post_process(self, dataset_config: dict[str, Any]):
        """Clean up angles and combine images internally."""
        # We set the samples with no info on inc_angle to 0 as its value, to simplify.
        self.dataframe[IcebergDataSet.ANGLE_KEY] = self.dataframe[
            IcebergDataSet.ANGLE_KEY
        ].replace("na", 0)
        self.dataframe[IcebergDataSet.ANGLE_KEY] = (
            self.dataframe[IcebergDataSet.ANGLE_KEY].astype(float).fillna(0.0)
        )
        print("Done cleaning up angle", flush=True)

        # Sets up a combined set of inputs containing each separate band, plus a combined image of both bands.
        x_band1 = np.array(self.dataframe[IcebergDataSet.BAND1_KEY])
        x_band2 = np.array(self.dataframe[IcebergDataSet.BAND2_KEY])
        square_x_band1 = np.array(
            [
                np.array(band)
                .astype(np.float32)
                .reshape(self.BAND_WIDTH, self.BAND_HEIGHT)
                for band in x_band1
            ]
        )
        square_x_band2 = np.array(
            [
                np.array(band)
                .astype(np.float32)
                .reshape(self.BAND_WIDTH, self.BAND_HEIGHT)
                for band in x_band2
            ]
        )
        self.x_combined_bands = np.concatenate(
            [
                square_x_band1[:, :, :, np.newaxis],
                square_x_band2[:, :, :, np.newaxis],
                ((square_x_band1 + square_x_band2) / 2)[:, :, :, np.newaxis],
            ],
            axis=-1,
        )

        print("Done post-processing data", flush=True)

    # Overriden (implemented abstract method).
    def get_model_inputs(self) -> list[SequenceLike]:
        """Returns the 2 inputs to be used: the combined bands and the angle."""
        return [
            self.x_combined_bands,
            np.array(self.dataframe[IcebergDataSet.ANGLE_KEY]),
        ]
