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

import json
from typing import Any, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd

from portend.analysis.predictions import Predictions
from portend.utils.logging import print_and_log

# Class definitions.
CLASS_NEGATIVE = 0
CLASS_POSITIVE = 1

DEFAULT_PREDICTION_PRECENTILE = 95
MEDIAN_PERCENTILE = 50
DEFAULT_NUM_POINTS = 50


def prep_metric_data(
    ids: Optional[npt.NDArray[Any]],
    predictions: Predictions,
    params: dict[str, Any],
) -> tuple[list[list[float]], npt.NDArray[Any], npt.NDArray[Any]]:
    """
    Loads confidence data from multiple Wildnav outputs about confidences, and structures it so it can be used in metrics.
    :return: probabilities for the dataset, labels from the predictions, and predictions for distance error.
    """
    print_and_log(
        "Executing prep function for Wildnav data, obtaining confidences needed for calculations of confidence-based metrics."
    )

    # Get the data for the results on each dataset.
    data_file = params.get("additional_data")
    data = pd.DataFrame.from_dict(predictions.get_additional_data(data_file))

    if "Confidence" not in data or "Confidence Invalid" not in data:
        raise RuntimeError(
            "Missing required confidence data in loaded additional data."
        )

    # Parse confidences.
    data["Parsed Confidence"] = data["Confidence"].apply(json.loads)
    data["Parsed Confidence Invalid"] = data["Confidence Invalid"].apply(
        json.loads
    )

    # Get the probabilities.
    num_points: Optional[int] = params.get("num_points")
    probabilities = _get_top_probabilities(data, num_points)

    # Obtain labels from data.
    distance_error_threshold: Optional[float] = params.get(
        "distance_error_threshold"
    )
    print_and_log(f"Configured distance error: {distance_error_threshold}")
    labels = _generate_labels(data, distance_error_threshold)

    # Obtain predictions from data.
    pred_error_percentile: Optional[float] = params.get(
        "prediction_error_percentile"
    )
    print_and_log(f"Configured prediction percentile: {pred_error_percentile}")
    matching_predictions = _get_predictions(data, pred_error_percentile)

    return probabilities, labels, matching_predictions


def _get_top_probabilities(
    data: pd.DataFrame, num_points: Optional[int]
) -> list[list[float]]:
    """
    Gets the confidence data from the additional data of a dataset loaded from a Wildnav CSV output file.
    It sorts the probabilities and takes the top num_points values for each confidence list.
    :return A numpy array with a list of a list of floats, including the top confidences for each classifier.
    """
    # Convert all data from JSON to dicts.
    print_and_log("Loading probabilities")

    if num_points is None:
        num_points = DEFAULT_NUM_POINTS

    # Go over all confidences, merge valid with invalid, and sort them.
    all_confidences: list[list[float]] = []
    for i in range(len(data)):
        confidence_list: list[float] = data["Parsed Confidence"][i]
        confidence_list.extend(data["Parsed Confidence Invalid"][i])
        confidence_list.sort(reverse=True)
        all_confidences.append(confidence_list)

    # For each list of sorted confidences, only take the top ones. Ignore confidences with no values.
    probabilities = [
        confidences[:num_points] for confidences in all_confidences
    ]

    return probabilities


def _generate_labels(
    data: pd.DataFrame, error_threshold: Optional[float] = None
) -> npt.NDArray[np.int_]:
    """Generate labels based on the distance error and the given error threshold."""
    if "Meters_Error" not in data:
        print_and_log("Missing meters error, won't generate labels.")
        return np.array([])

    if error_threshold is None:
        if "Matched" not in data:
            raise RuntimeError(
                "Missing required matched data in loaded additional data."
            )

        error_threshold = np.percentile(
            data.loc[data["Matched"] == True]["Meters_Error"],  # noqa
            MEDIAN_PERCENTILE,
        )

    labels = [
        CLASS_POSITIVE if i <= error_threshold else CLASS_NEGATIVE
        for i in data["Meters_Error"]
    ]
    return np.array(labels)


def _get_predictions(
    data: pd.DataFrame,
    percentile: Optional[float] = None,
) -> npt.NDArray[np.int_]:
    """Get the predictions are based on the distribution of meters error for the region for the given threshold."""
    if "Meters_Error" not in data:
        print_and_log("Missing meters error, won't generate predictions.")
        return np.array([])

    if percentile is None:
        percentile = DEFAULT_PREDICTION_PRECENTILE
    error_threshold = np.percentile(data["Meters_Error"], percentile)
    preds = [
        CLASS_POSITIVE if i < error_threshold else CLASS_NEGATIVE
        for i in data["Meters_Error"]
    ]
    return np.array(preds)
