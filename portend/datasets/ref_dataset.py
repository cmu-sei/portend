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

import secrets
import typing
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from portend.datasets.dataset import DataSet
from portend.utils.typing import SequenceLike


class RefDataSet(DataSet):
    """A dataset referencing the ids of another dataset."""

    ORIGINAL_ID_KEY = "original_id"
    x_original_ids = np.empty(0, str)

    SAMPLE_GROUP_ID_KEY = "sample_group_id"
    sample_group_ids = np.empty(0, str)

    # Overriden to add additional internal data.
    def as_dataframe(self, only_ids: bool = False) -> pd.DataFrame:
        """Adds internal data to a new dataframe."""
        dataset_df = super().as_dataframe(only_ids)
        if not only_ids:
            dataset_df[RefDataSet.ORIGINAL_ID_KEY] = self.x_original_ids
            dataset_df[RefDataSet.SAMPLE_GROUP_ID_KEY] = self.sample_group_ids
        return dataset_df

    # Overriden to add post processing.
    def post_process(self, dataset_config: dict[str, typing.Any]):
        """Prepares dataset after it has been loaded."""
        self.x_original_ids = np.array(
            self.dataframe[RefDataSet.ORIGINAL_ID_KEY]
        )
        self.sample_group_ids = np.array(
            self.dataframe[RefDataSet.SAMPLE_GROUP_ID_KEY]
        )
        self.dataframe.drop(RefDataSet.ORIGINAL_ID_KEY, axis=1, inplace=True)
        self.dataframe.drop(
            RefDataSet.SAMPLE_GROUP_ID_KEY, axis=1, inplace=True
        )
        print("Done loading original ids", flush=True)

    # Overriden so that this class can be instantiated, but this method should not be used for this class type.
    def get_model_inputs(self) -> list[SequenceLike]:
        return []

    def get_original_ids(self) -> npt.NDArray[Any]:
        return self.x_original_ids

    def get_num_sample_groups(self) -> int:
        """Gets the number of sample groups in this dataset."""
        return int(np.unique(self.sample_group_ids).size)

    def get_sample_group_size(self) -> int:
        """Gets the size of each sample group. All are assumed to have the same size."""
        _, counts = np.unique(self.sample_group_ids, return_counts=True)
        return int(counts[0])

    def new_empty_sample(self) -> list[typing.Any]:
        """Returns a new sample, with random id and no timestamp."""
        new_id = secrets.token_hex(10)
        return [new_id, 0]

    def create_from_lists(
        self,
        samples: list[Any],
        original_ids: list[Any],
        sample_group_ids: list[Any],
    ):
        """Adds multiple original ids by reference."""
        self.dataframe = pd.DataFrame(
            samples, columns=[self.id_key, self.timestamp_key]
        )
        self.x_original_ids = np.array(original_ids)
        self.sample_group_ids = np.array(sample_group_ids)

    def create_from_reference(
        self,
        base_dataset: DataSet,
        output_dataset_class: typing.Type[DataSet],
        dataset_config: dict[str, Any],
    ) -> DataSet:
        """Creates a new dataset by getting the full samples of a reference from the base dataset."""
        # Gather all complete samples from the refs and full data.
        samples = []
        for idx, original_id in enumerate(self.get_original_ids()):
            # Only show print update every 500 ids.
            if idx % 500 == 0:
                print(
                    f"Finished preparing {idx} samples out of {self.get_number_of_samples()}",
                    flush=True,
                )

            # Get the full sample, but replace the timestamp (if any) with the one from the reference dataset.
            full_sample = base_dataset.get_sample_by_id(original_id)
            full_sample[self.timestamp_key] = self.dataframe.loc[idx][
                self.timestamp_key
            ]
            samples.append(full_sample)

        # Add all samples to create the new dataset.
        new_dataset: DataSet = output_dataset_class()
        new_dataset.set_samples(samples)

        # Set all keys and post-process if needed.
        new_dataset.set_id_key(base_dataset.id_key)
        new_dataset.set_model_input_key(base_dataset.input_key)
        new_dataset.set_model_output_key(base_dataset.output_key)
        new_dataset.post_process(dataset_config)

        return new_dataset
