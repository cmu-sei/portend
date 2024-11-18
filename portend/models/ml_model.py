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
from abc import ABCMeta, abstractmethod
from typing import Any, Type

import sklearn.model_selection as skm

from portend.datasets.dataset import DataSet
from portend.training.training_set import TrainingSet
from portend.utils.module import load_class

DEFAULT_MODEL_CLASS = "portend.models.keras_model.KerasModel"


class MLModel(metaclass=ABCMeta):
    """Base class for ML models to be used with the system."""

    def set_model(self, model: Any):
        """Sets a generic model."""
        self.model = model

    def load_from_file(self, model_filename: str):
        """Loads a model from a given file."""
        raise NotImplementedError("Load method not implemented")

    @abstractmethod
    def predict(self, input: Any) -> tuple[Any, dict[str, dict[str, Any]]]:
        """Runs the model to get predictions, and any additional data."""
        raise NotImplementedError("Predict method not implemented")

    def load_additional_params(self, data: dict[str, Any]):
        """Optional function, in case additional params needs to be loaded for the model."""
        return

    ####################################################################################
    # The following methods only need to be implemented if the trainer is used.
    # They should be implemented once per type of library used for a model, not for each extension.
    ####################################################################################
    def save_to_file(self, model_filename: str):
        raise NotImplementedError("Save method not implemented")

    def train(self, training_set: TrainingSet, config_params: dict[str, Any]):
        raise NotImplementedError("Train method not implemented")

    def evaluate(
        self,
        evaluation_input: list[Any],
        evaluation_output: list[Any],
        config_params: dict[str, Any],
    ):
        raise NotImplementedError("Evaluate method not implemented")

    ####################################################################################
    # The following methods only need to be implemented/overriden if the trainer is used.
    # They should be implemented for each extension model as needed.
    ####################################################################################

    def create_model(self, *args, **kwargs):
        """Creates the model internally."""
        raise NotImplementedError("Create model method not implemented")

    def split_data(
        self, dataset: DataSet, validation_percentage: float
    ) -> TrainingSet:
        """Split training set into train and validation (validation_percentage% to actually train)"""
        model_inputs = dataset.get_model_inputs()
        output = dataset.get_model_output()

        lists = [input for input in model_inputs]
        lists.append(output)
        split_results = skm.train_test_split(
            *lists, random_state=42, test_size=validation_percentage
        )
        print("Done splitting validation data from train data", flush=True)
        print("Num results: " + str(len(split_results)))

        # Remove the output train/val to get the inputs.
        training_set = TrainingSet()
        split_inputs = split_results[0 : len(split_results) - 2]
        if len(model_inputs) > 1:
            # Even positions are train, odds are test.
            training_set.x_train = [
                input for pos, input in enumerate(split_inputs) if pos % 2 == 0
            ]
            training_set.x_validation = [
                input for pos, input in enumerate(split_inputs) if pos % 2 != 0
            ]
        else:
            training_set.x_train = split_inputs[0]
            training_set.x_validation = split_inputs[1]

        # Get the last couple of results for the split output.
        split_outputs = split_results[-2 : len(split_results)]
        training_set.y_train = split_outputs[0]
        training_set.y_validation = split_outputs[1]
        training_set.num_train_samples = len(training_set.y_train)
        training_set.num_validation_samples = (
            len(training_set.y_validation)
            if training_set.y_validation is not None
            else 0
        )
        return training_set

    def get_fold_data(
        self, dataset: DataSet, train_index: int, test_index: int
    ) -> TrainingSet:
        """Prepares a training set for the given dataset and indexes"""
        model_inputs = dataset.get_model_inputs()
        output = dataset.get_model_output()

        training_set = TrainingSet()

        # If multiple inputs, slice them in an array. If not, just put the first/only input directly.
        if len(model_inputs) > 1:
            training_set.x_train = [
                input[train_index] for input in model_inputs
            ]
            training_set.x_validation = [
                input[test_index] for input in model_inputs
            ]
        else:
            training_set.x_train = model_inputs[0][train_index]
            training_set.x_validation = model_inputs[0][test_index]

        training_set.y_train = output[train_index]
        training_set.y_validation = output[test_index]
        training_set.num_train_samples = len(training_set.y_train)
        training_set.num_validation_samples = (
            len(training_set.y_validation)
            if training_set.y_validation is not None
            else 0
        )
        return training_set


def load_model(model_config: dict[str, str]) -> MLModel:
    """Load model from class name, and from file if provided."""
    model_filename = model_config.get("model_file")
    model_class_name = (
        model_config.get("model_class")
        if "model_class" in model_config
        else DEFAULT_MODEL_CLASS
    )

    # Create an instance of the model class.
    model_class: Type[MLModel] = typing.cast(
        Type[MLModel], load_class(model_class_name)
    )
    model_instance: MLModel = model_class()

    # Only if filename is provided, load the model itself from the provided file.
    if model_filename is not None:
        model_instance.load_from_file(model_filename)

    # If a model class implements this, allow it to process additional params if needed.
    model_instance.load_additional_params(model_config)

    return model_instance
