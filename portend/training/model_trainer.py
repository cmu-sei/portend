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

from sklearn.model_selection import KFold

from portend.datasets.dataset import DataSet
from portend.models.ml_model import MLModel
from portend.utils.logging import print_and_log


class ModelTrainer:
    """Has functionalities to train a model"""

    def __init__(self, model_instance: MLModel, config_params: dict[str, Any]):
        self.model_instance = model_instance
        self.config_params = config_params
        self.evaluation_input: Optional[list[Any]] = None
        self.evaluation_output: Optional[list[Any]] = None

    def cross_validate(self, full_dataset: DataSet, num_folds: int = 5):
        """k-fold cross-validation to check how model is performing by selecting different sets to train/validate."""

        # Define the K-fold Cross Validator
        print_and_log("CROSS VALIDATION")
        kfold = KFold(n_splits=num_folds, shuffle=True)

        # K-fold Cross Validation model evaluation
        fold_no = 1
        for train_index, test_index in kfold.split(
            full_dataset.get_model_inputs()[0], full_dataset.get_model_output()
        ):
            # Generate a print
            print(
                "------------------------------------------------------------------------"
            )
            print_and_log(f"Training for fold {fold_no} ...")

            training_set = self.model_instance.get_fold_data(
                full_dataset, train_index, test_index
            )

            # Fit data to model
            print_and_log(
                f"Training fold samples: {training_set.num_train_samples}"
            )
            self.model_instance.train(training_set, self.config_params)

            # Generate generalization metrics
            print_and_log(
                f"Evaluation fold samples: {training_set.num_validation_samples}"
            )
            self.evaluate(training_set.x_validation, training_set.y_validation)

            # Increase fold number
            fold_no = fold_no + 1

        print_and_log("Done with cross-validation!")

    def split_and_train(self, dataset_instance: DataSet):
        """Splits a dataset and trains the configured model, returning it."""
        training_set = self.model_instance.split_data(
            dataset_instance, self.config_params.get("validation_size", 30)
        )
        print_and_log(
            f"Dataset samples {dataset_instance.get_number_of_samples()}, "
            f"training samples: {len(training_set.x_train[0])}, "
            f"validation samples: {len(training_set.x_validation[0] if training_set.x_validation is not None else 0)}"
        )

        self.model_instance.train(training_set, self.config_params)

        # Store evaluation input/outputs as the validation split, in case evaluation is done later.
        self.evaluation_input = training_set.x_validation
        self.evaluation_output = training_set.y_validation

    def evaluate(self, evaluation_input=None, evaluation_output=None):
        """Does an evaluation."""
        print_and_log("EVALUATION")
        print("Starting evaluation", flush=True)
        if self.evaluation_input is not None:
            evaluation_input = self.evaluation_input
        if self.evaluation_output is not None:
            evaluation_output = self.evaluation_output
        if evaluation_input is None or evaluation_output is None:
            raise Exception(
                "Evaluation input or output not passed properly to evaluate."
            )

        self.model_instance.evaluate(
            evaluation_input, evaluation_output, self.config_params
        )
