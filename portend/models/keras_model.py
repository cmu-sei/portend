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

import tensorflow.keras as keras
import tensorflow.keras.callbacks as tfcb

from portend.models.ml_model import MLModel
from portend.training.training_set import TrainingSet
from portend.utils.logging import print_and_log
from portend.utils.typing import SequenceLike

DEFAULT_EPOCCHS = 10
DEFAULT_BATCH_SIZE = 32


class KerasModel(MLModel):
    """Base class for Keras-based ML models to be used with the system."""

    # Implemented.
    def save_to_file(self, model_filename: str):
        self.model.save(model_filename)

    # Implemented.
    def load_from_file(self, model_filename: str):
        self.model = keras.models.load_model(model_filename)
        self.model.summary()

    # Implemented.
    def predict(
        self, input: Any
    ) -> tuple[SequenceLike, dict[str, dict[str, Any]]]:
        raw_predictions = self.model.predict(input).flatten()
        print_and_log(f"Predictions shape: {raw_predictions.shape}")
        return raw_predictions, {}

    # Implemented.
    def train(self, training_set: TrainingSet, config_params: dict[str, Any]):
        """Train. Requires at least the following config_params: "epochs" and "batch_size" """
        print_and_log("TRAINING")

        self.create_model()

        epochs = config_params.get("epochs", DEFAULT_EPOCCHS)
        batch_size = config_params.get("batch_size", DEFAULT_BATCH_SIZE)
        print_and_log(
            f"Starting training with hyper parameters: epochs: {epochs}, batch size: {batch_size}"
        )

        validation_data = None
        callbacks = None
        if training_set.has_validation():
            print_and_log("Validation data found")
            validation_data = (
                training_set.x_validation,
                training_set.y_validation,
            )
            callbacks = self.get_callbacks(patience=5)

        history = self.model.fit(
            training_set.x_train,
            training_set.y_train,
            epochs=epochs,
            validation_data=validation_data,
            batch_size=batch_size,
            callbacks=callbacks,
        )
        print_and_log(
            f'Final training result ({len(history.history.get("loss"))} epochs): '
            f'loss: {history.history.get("loss")[-1]}, '
            f'accuracy: {history.history.get("accuracy")[-1]}'
        )
        if training_set.has_validation():
            print_and_log(
                f'Validation: val_loss: {history.history.get("val_loss")[-1]}, '
                f'val_accuracy: {history.history.get("val_accuracy")[-1]}'
            )

        print("Done training!", flush=True)

    # Implemented.
    def evaluate(
        self,
        evaluation_input: list[Any],
        evaluation_output: list[Any],
        config_params: dict[str, Any],
    ):
        """Does an evaluation."""
        batch_size = config_params.get("batch_size", DEFAULT_BATCH_SIZE)
        scores = self.model.evaluate(
            evaluation_input, evaluation_output, batch_size=batch_size
        )
        print(f"Done! Evaluation loss and acc: {scores}")
        print_and_log(
            f"Score: {self.model.metrics_names[0]} of {scores[0]}; "
            f"{self.model.metrics_names[1]} of {scores[1] * 100}%"
        )

    @staticmethod
    def get_callbacks(patience=2):
        """Gets helper callbacks to save checkpoints and allow early stopping when needed."""
        file_path = ".model_weights.hdf5"
        es = tfcb.EarlyStopping("val_loss", patience=patience, mode="min")
        msave = tfcb.ModelCheckpoint(file_path, save_best_only=True)
        return [es, msave]
