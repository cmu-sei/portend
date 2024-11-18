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
from typing import Any, Dict, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd

from portend.utils import dataframe_helper
from portend.utils import files as file_utils
from portend.utils.typing import SequenceLike

############################
# DataSet class.
############################


class DataSet:
    """A basic dataset, containing ids, optional timestamp, and additional data as dataframes."""

    DEFAULT_ID_KEY = "id"
    DEFAULT_TIMESTAMP_KEY = "timestamp"

    # Generic way to store data, for simple datasets.
    dataframe = pd.DataFrame()

    # String keys to get basic data out of the dataframe.
    id_key: str = DEFAULT_ID_KEY
    timestamp_key: str = DEFAULT_TIMESTAMP_KEY
    input_key: Optional[str] = None
    output_key: Optional[str] = None

    ############################
    # ID methods.
    ############################

    def set_id_key(self, ids_key: str):
        """Sets the id key."""
        self.id_key = ids_key

    def get_ids(self) -> npt.NDArray[Any]:
        """Returns the dataset ids."""
        return np.array(self.dataframe[self.id_key])

    def get_id_position(self, id_to_find: str) -> int:
        """Gets the position of a given id"""
        position_info = np.where(self.get_ids() == id_to_find)
        if len(position_info[0]) == 0:
            raise Exception(f"Id {id_to_find} not found")
        return int(position_info[0][0])

    ############################
    # Timestamp methods.
    ############################

    def set_timestamp_key(self, timestamps_key: str):
        """Sets the timestamps key."""
        self.timestamp_key = timestamps_key

    def get_timestamps(self) -> npt.NDArray[np.int_]:
        """Returns the timestamps, as Unix timestamps."""
        return np.array(self.dataframe[self.timestamp_key])

    def set_timestamps(self, timestamps: npt.NDArray[np.int_]):
        """Sets the timestamps, as Unix timestamps."""
        self.dataframe[self.timestamp_key] = timestamps

    def has_timestamps(self) -> bool:
        """Check if this dataset has timestamps."""
        return self.timestamp_key in self.dataframe.columns

    ##################################
    # Data/sample handling methods.
    ##################################

    def get_number_of_samples(self) -> int:
        """Gets the current size in num of samples."""
        return len(self.dataframe.index)

    def get_sample_by_id(self, id_to_find: str) -> dict[str, Any]:
        """Returns a sample given its id."""
        position = self.get_id_position(id_to_find)
        return self.get_sample(position)

    def get_sample(self, position: int) -> dict[str, Any]:
        """Returns a sample associated to this id as a dictionary."""
        if position < self.get_number_of_samples():
            return typing.cast(
                Dict[str, Any], self.dataframe.loc[position].to_dict()
            )
        else:
            return {}

    def set_samples(self, samples: list[dict[str, Any]]):
        """Sets the given samples."""
        if len(samples) > 0:
            self.dataframe = pd.DataFrame(
                typing.cast(list[Any], samples),
                columns=typing.cast(list[Any], samples[0].keys()),
            )

    def get_samples(self):
        """Gets the samples."""
        return self.dataframe.to_dict("records")

    ########################################
    # Setup and dataframe serialization.
    ########################################

    def as_dataframe(self, only_ids: bool = False) -> pd.DataFrame:
        """Returns the dataset as as dataframe."""
        if only_ids:
            # If this option is selected, we only want to return ids, so we drop everything else.
            columns = {self.id_key: self.dataframe[self.id_key]}
            dataset_df = pd.DataFrame().assign(**columns)
        else:
            dataset_df = self.dataframe.copy(deep=True)

        return dataset_df

    def from_dataframe(self, dataframe: pd.DataFrame):
        """Loads ids and times from a dataframe file into this object.
        convert_to_timestamp should be a function that converts from whatever format the dataset has its datetime,
        to Unix timestamp."""
        self.dataframe = dataframe
        print("Creating dataset from dataframe. Sample row: ")
        print(self.dataframe.head(1))

    # May be overriden if derived dataset has extra data attributes.
    def clone(self, cloned_dataset: DataSet) -> DataSet:
        """Clones into the provided dataset."""
        cloned_dataset.id_key = self.id_key
        cloned_dataset.timestamp_key = self.timestamp_key
        cloned_dataset.input_key = self.input_key
        cloned_dataset.output_key = self.output_key

        cloned_dataset.from_dataframe(self.as_dataframe())
        return cloned_dataset

    ###############################
    # File loading/saving mehods.
    ###############################

    def load_from_file(
        self,
        dataset_filename: Optional[str],
        dataset_config: dict[str, Any],
        convert_to_timestamp: Optional[
            typing.Callable[[SequenceLike], npt.NDArray[np.int_]]
        ] = None,
    ):
        """Loads ids and times from a JSON file into this object. Stores the rest as a dataframe.
        convert_to_timestamp should be a function that converts from whatever format the dataset has its datetime,
        to Unix timestamp."""
        if dataset_filename is None:
            raise Exception("Can't load data from file, no filename provided")

        # Load all keys. Input and Output keys can be None by default.
        self.id_key = dataset_config.get(
            "dataset_id_key", DataSet.DEFAULT_ID_KEY
        )
        self.timestamp_key = dataset_config.get(
            "dataset_timestamp_key", DataSet.DEFAULT_TIMESTAMP_KEY
        )
        self.input_key = dataset_config.get("dataset_input_key")
        self.output_key = dataset_config.get("dataset_output_key")

        # Load from file.
        dataset_df = dataframe_helper.load_dataframe_from_file(dataset_filename)
        self.from_dataframe(dataset_df)

        # Set up timestamps, cleaning up null ones and converting it to specified internal format if needed.
        try:
            if self.has_timestamps():
                if convert_to_timestamp is None:
                    converted_timestamps = np.array(
                        self.dataframe[self.timestamp_key]
                    )
                else:
                    converted_timestamps = convert_to_timestamp(
                        np.array(self.dataframe[self.timestamp_key])
                    )
                self.set_timestamps(converted_timestamps)
        except KeyError as ex:
            raise Exception(
                f"Could not setup timestamps from dataset: {type(ex).__name__}: {str(ex)}"
            )

        print("Done setting up keys and timestamps", flush=True)

        # Call optional post process function, if needed to process loaded data.
        self.post_process(dataset_config)

    # May be optionally overriden to post-process data.
    def post_process(self, dataset_config: dict[str, Any]):
        """Only needs ot be extended if there is post-processing that is needed after loading from a file."""
        return

    def save_to_file(self, output_filename: str):
        """Stores Numpy arrays with a dataset into a JSON file."""
        file_utils.create_folder_for_file(output_filename)
        dataset_df = self.as_dataframe()
        dataframe_helper.save_dataframe_to_file(dataset_df, output_filename)

    ###############################
    # Model input/output methods.
    ###############################

    def set_model_input_key(self, input_key: Optional[str]):
        """Sets the input key to be used, by default, if the input is a simple column from the dataset."""
        self.input_key = input_key

    # Will usually need to be overriden to return model info.
    def get_model_inputs(self) -> list[SequenceLike]:
        """Returns the inputs to be used as list of numpy arrays. Each item in the list in an array of features. Has to contain at least one array of inputs."""
        if self.input_key is not None:
            if self.input_key not in self.dataframe:
                raise Exception(
                    f"Model input key '{self.input_key}' is not in dataset."
                )
            return [np.array(self.dataframe[self.input_key])]
        else:
            raise Exception(
                "Input key has not been set, can't return model input."
            )

    def set_model_output_key(self, output_key: Optional[str]):
        """Sets the output key to be used."""
        self.output_key = output_key

    def get_model_output(self) -> SequenceLike:
        """Has to return a numpy array with the output."""
        if self.output_key is not None:
            if self.output_key not in self.dataframe:
                raise Exception(
                    f"Model output key '{self.output_key}' is not in dataset."
                )
            return np.array(list(self.dataframe[self.output_key]))
        else:
            raise Exception(
                "Output key has not been set, can't return model output."
            )

    def set_model_output(self, new_output: SequenceLike):
        """Gets a numpy array and sets it as the output."""
        if self.output_key is not None:
            self.dataframe[self.output_key] = new_output
        else:
            raise Exception("Can't set output, output key has not been set!")
