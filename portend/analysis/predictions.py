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
from typing import Any, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd

from portend.analysis.accuracy import ClassificationAccuracy
from portend.utils import dataframe_helper
from portend.utils import files as file_utils
from portend.utils.logging import print_and_log
from portend.utils.typing import SequenceLike

JSON_EXT = ".json"
ADD_DATA_FILE_SUFFIX = "_add_data"


def build_predictions_object(
    raw_predictions: Optional[SequenceLike] = None,
    expected_output: Optional[SequenceLike] = None,
    additional_data: Optional[dict[str, dict[str, Any]]] = None,
    class_params: Optional[dict[str, Any]] = None,
) -> Predictions:
    """Sets up a Predictions object with the given params."""
    predictions = Predictions()
    if raw_predictions is not None:
        print_and_log(f"Got predictions: {len(raw_predictions)}")
        predictions.store_predictions(raw_predictions)
    if expected_output is not None:
        predictions.store_expected_results(expected_output)
    if additional_data is not None:
        print_and_log(f"Got additional data: {additional_data.keys()}")
        predictions.store_additional_data(additional_data)

    # If we want to classify predictions, update result to do it.
    if class_params is not None:
        predictions = ClassPredictions.from_predictions(
            predictions, class_params
        )

    return predictions


class Predictions:
    """Class to store and handle prediction results."""

    TRUTH_KEY = "truth"
    PREDICTIONS_KEY = "prediction"
    ADDITIONAL_DATA = "additional_data"

    raw_predictions: npt.NDArray[Any] = np.empty(1)
    expected_results: npt.NDArray[Any] = np.empty(1)
    additional_data: dict[str, dict[str, Any]] = {}

    def store_expected_results(self, expected_output: SequenceLike):
        """Stores expected results and confusion matrix."""
        self.expected_results = np.array(expected_output)

    def get_expected_results(self) -> npt.NDArray[Any]:
        """Returns the ground truth classification."""
        return self.expected_results

    def store_additional_data(self, additional_data: dict[str, dict[str, Any]]):
        """Stores additional data about the predictions."""
        self.additional_data = additional_data

    def get_additional_data(self, key: Optional[str] = None) -> dict[str, Any]:
        """Return the additional data specified by the given key."""
        if key is None:
            return self.additional_data
        if key not in self.additional_data:
            raise Exception(f"Key '{key}' was not found in additional data.")
        else:
            return self.additional_data[key]

    def store_predictions(self, raw_predictions: SequenceLike):
        """Stores raw predicitons."""
        self.raw_predictions = np.array(raw_predictions)

    def get_predictions(self) -> npt.NDArray[Any]:
        """Returns the raw predictions."""
        return self.raw_predictions

    def create_slice(self, starting_idx: int, size: int) -> Predictions:
        """Creates a new object of this type with a slice of the results in this one."""
        if size == 0:
            raise Exception("Can't create a slice of size 0.")
        sliced_predictions = Predictions()
        sliced_predictions.store_additional_data(self.get_additional_data())
        sliced_predictions.store_expected_results(
            self.get_expected_results()[starting_idx : starting_idx + size]
        )
        sliced_predictions.store_predictions(
            self.get_predictions()[starting_idx : starting_idx + size]
        )
        return sliced_predictions

    def as_dataframe(
        self, dataframe: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Returns a dataframe with this predictions object, adding to existing argument if received."""
        if dataframe is None:
            dataframe = pd.DataFrame()
        else:
            dataframe = dataframe.copy(deep=True)

        dataframe[self.TRUTH_KEY] = pd.Series(list(self.get_expected_results()))
        dataframe[self.PREDICTIONS_KEY] = pd.Series(
            list(self.get_predictions())
        )
        return dataframe

    def save_to_file(
        self, output_filepath: str, ids_df: Optional[pd.DataFrame] = None
    ):
        """Saves this prediction object to file"""
        # First save predictions.
        predictions_df = self.as_dataframe(ids_df)
        dataframe_helper.save_dataframe_to_file(predictions_df, output_filepath)

        # Separately save additional data.
        add_data_output_filename = self.get_add_data_filename(output_filepath)
        add_data = self.get_additional_data()
        if len(add_data) > 0:
            file_utils.save_dict_to_json_file(
                self.get_additional_data(),
                add_data_output_filename,
                data_name=self.ADDITIONAL_DATA,
            )

    @staticmethod
    def get_add_data_filename(predictions_filepath: str) -> str:
        """Returns the filename and path of the file used for additional data, based on the given main predictions file."""
        return predictions_filepath.replace(
            JSON_EXT, ADD_DATA_FILE_SUFFIX + JSON_EXT
        )

    @staticmethod
    def load_from_file(
        predictions_filename: str, class_params: Optional[dict[str, Any]] = None
    ) -> Predictions:
        """Loads predictions info from a file."""
        # Load main predictions data.
        predictions_df = dataframe_helper.load_dataframe_from_file(
            predictions_filename
        )

        # Load additional data, if any.
        add_data: Optional[dict[str, Any]] = None
        try:
            add_data_filename = Predictions.get_add_data_filename(
                predictions_filename
            )
            add_data = file_utils.load_json_file_to_dict(
                add_data_filename, Predictions.ADDITIONAL_DATA
            )
        except IOError:
            print_and_log("Additional data file not found, ignoring.")

        return build_predictions_object(
            raw_predictions=np.array(
                predictions_df[Predictions.PREDICTIONS_KEY]
            ),
            expected_output=np.array(predictions_df[Predictions.TRUTH_KEY]),
            additional_data=add_data if add_data is not None else None,
            class_params=class_params,
        )


class ClassPredictions(Predictions):
    """A Predictions class for classification problems."""

    RAW_PREDICTIONS_KEY = "raw_prediction"

    DEFAULT_THRESHOLD = 0.5
    DEFAULT_POSITIVE_CLASS = 1
    DEFAULT_LABELS = [0, 1]

    class_params: dict[str, Any] = {}
    classification_threshold: float = 0
    classified_predictions: npt.NDArray[Any] = np.empty(1)
    accuracy: ClassificationAccuracy = ClassificationAccuracy()

    @staticmethod
    def from_predictions(
        predictions: Predictions, class_params: dict[str, Any]
    ) -> ClassPredictions:
        # TODO: deep clone?
        classified_predictions = ClassPredictions()
        classified_predictions.class_params = class_params
        classified_predictions.store_additional_data(
            predictions.get_additional_data()
        )
        classified_predictions.store_expected_results(
            predictions.get_expected_results()
        )
        classified_predictions.store_predictions(predictions.get_predictions())
        return classified_predictions

    @staticmethod
    def classify(
        predictions: npt.NDArray[Any], threshold: float
    ) -> npt.NDArray[Any]:
        """Turns raw predictions into actual classification. Only 1/0 for now."""
        # TODO: Add support for more complex classifications, and for using resulting labels from class_params here as well.
        return np.where(predictions > threshold, 1, 0)

    def classify_raw_predictions(self):
        """Generates classified predictions based on the raw ones."""
        if len(self.class_params) == 0:
            raise RuntimeError(
                "Can't classify predictions, classification params have not been set."
            )

        self.classification_threshold = typing.cast(
            float, self.class_params.get("threshold", self.DEFAULT_THRESHOLD)
        )
        self.classified_predictions = self.classify(
            self.raw_predictions, self.classification_threshold
        )

    # Overriden
    def store_predictions(self, raw_predictions: SequenceLike):
        """Stores already raw predictions, and also classifies them."""
        super().store_predictions(raw_predictions)
        self.classify_raw_predictions()

    # Overriden
    def get_predictions(self) -> npt.NDArray[Any]:
        """Returns the classified predictions."""
        return self.classified_predictions

    def get_raw_predictions(self) -> npt.NDArray[Any]:
        """User to access the raw predictions from base class, if needed."""
        return super().get_predictions()

    def calculate_accuracy(self) -> float:
        """Calculates accuracy."""
        # TODO: get Labels and Positive case from class_params and pass as arguments to accuracy.
        return self.accuracy.calculate_accuracy(
            self.expected_results,
            self.classified_predictions,
            labels=self.DEFAULT_LABELS,
            positive_class=self.DEFAULT_POSITIVE_CLASS,
        )

    # Overriden.
    def create_slice(self, starting_idx: int, size: int) -> ClassPredictions:
        """Creates a new object of this type with a slice of the results in this one."""
        sliced_predictions = super().create_slice(starting_idx, size)
        class_predictions = ClassPredictions.from_predictions(
            sliced_predictions, self.class_params
        )
        return class_predictions

    # Overriden.
    def as_dataframe(
        self, dataframe: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Returns a dataframe with this predictions object, adding to existing argument if received."""
        dataframe = super().as_dataframe(dataframe)
        dataframe[self.RAW_PREDICTIONS_KEY] = pd.Series(
            list(self.get_raw_predictions())
        )
        return dataframe
