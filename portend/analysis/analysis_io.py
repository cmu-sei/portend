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

from portend.analysis.predictions import Predictions
from portend.datasets.dataset import DataSet
from portend.utils import files as file_utils
from portend.utils.logging import print_and_log


def save_updated_dataset(
    updated_dataset: DataSet,
    predictions: Predictions,
    output_filename: Optional[str],
):
    """Saves a dataset to a JSON file, adding the given predictions first."""
    if output_filename is None:
        raise Exception("No output dataset filename was provided")

    updated_dataset.set_model_output(predictions.get_predictions())
    updated_dataset.save_to_file(output_filename)


def save_predictions(
    full_dataset: DataSet,
    predictions: Predictions,
    output_filename: Optional[str],
):
    """Saves the ids, predictions and expected results into a JSON file."""
    if output_filename is None:
        raise Exception("No predictions output filename was provided")
    print_and_log("Saving predictions to file")
    ids_df = full_dataset.as_dataframe(only_ids=True)
    predictions.save_to_file(output_filename, ids_df)


def save_metrics(metrics: dict[Any, Any], metrics_filename: str):
    """Stores the given metrics to an output."""
    if len(metrics) == 0:
        print_and_log("No metrics to store to file.")
        return

    file_utils.save_dict_to_json_file(
        metrics, metrics_filename, data_name="metrics"
    )


def load_metrics(metrics_file_path: str) -> dict[str, Any]:
    """Loads metrics data from a file into a dict."""
    return file_utils.load_json_file_to_dict(metrics_file_path, "metrics")


def load_predictions(
    predictions_file_path: str,
    class_params: Optional[dict[str, Any]],
) -> Predictions:
    """Loads stored predictions."""
    print_and_log(f"Loading prediction data from file {predictions_file_path}")
    return Predictions.load_from_file(predictions_file_path, class_params)
