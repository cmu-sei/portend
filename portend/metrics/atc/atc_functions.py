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
import numpy.typing as npt
import pandas as pd

from portend.metrics.atc.ATC_code import ATC_helper
from portend.utils.logging import print_and_log


def calculate_atc_threshold(
    probabilities: list[Any],
    labels: npt.NDArray[Any],
    predictions: npt.NDArray[Any],
) -> float:
    """
    Calculates the ATC threshold.
    :param probabilities: A numpy array with a list of float probabilities for each sample.
    :param labels: A numpy array with the label for each sample.
    """
    print_and_log("Calculating ATC threshold")
    atc_scores = _calculate_atc_scores(probabilities)

    # Filter all samples with a zero score.
    labels = _filter_out_zeros(labels, atc_scores)
    predictions = _filter_out_zeros(predictions, atc_scores)
    filtered_atc_scores = _filter_out_zeros(atc_scores, atc_scores)

    # Get and return the threshold.
    _, atc_threshold = ATC_helper.find_ATC_threshold(
        filtered_atc_scores, labels == predictions
    )
    threshold = float(atc_threshold)
    print_and_log(f"ATC predicted threshold {threshold}")
    return threshold


def _calculate_atc_scores(probabilities: list[Any]) -> npt.NDArray[Any]:
    """
    Calculates the ATC scores for the given confidences.
    :param probabilities: A numpy array containing a list of float probabilities for each sample.
    """
    if _is_list_empty(probabilities):
        raise RuntimeError("Empty list of probabilities was received.")

    scores = ATC_helper.get_entropy(pd.DataFrame(probabilities))
    return np.array(scores)


def _filter_out_zeros(
    array: npt.NDArray[Any], array_with_zeros: npt.NDArray[Any]
) -> npt.NDArray[Any]:
    """Filters out elements in the first array that are zero in the second array."""
    filtered: npt.NDArray[Any] = array[array_with_zeros != 0]
    return filtered


def calculate_atc_accuracy(
    probabilities: list[Any],
    avg_atc_threshold: float,
    window_size: int = 0,
    accumulated_scores: list[Any] = [],
) -> float:
    """
    Calculates the ATC accuracy.

    :param probabilities: A numpy array containing a list of probabilities for each sample.
    :param avg_atc_threshold: The ATC threshold to use for this calculation.
    :param window_size: Whether to use a sliding window or not. A value of 0 means no window.
    :param accumulated_scores: A list of accumulated scores, used only if sliding window is enabled.

    :return: The ATC accuracy.
    """
    print_and_log("Calculating ATC accuracy")
    print_and_log(f"Using ATC threshold: {avg_atc_threshold}")
    print_and_log(f"Probabilities: {probabilities}")

    # If we have no valid probabilities, raise error.
    if _is_list_empty(probabilities):
        raise RuntimeError("Empty list of probabilities was received.")

    # If we only have 0, raise error.
    if _are_all_values_zero(probabilities):
        raise RuntimeError("All probabilities are 0, can't calculate accuracy.")

    # First calculate the scores.
    test_scores = _calculate_atc_scores(probabilities)
    print_and_log(f"Scores: {test_scores.tolist()}")

    # Check if we want to use a sliding window.
    if window_size > 0:
        # Accumulate the new score so we can have a sliding window.
        accumulated_scores.extend(test_scores)

        # Check if we have enough samples for a window, and if so, make the window.
        if len(accumulated_scores) >= window_size:
            test_scores = np.array(accumulated_scores[-window_size:])
            print_and_log(f"With prev scores: {test_scores.tolist()}")
        else:
            raise RuntimeError(
                f"Not enough score values ({len(accumulated_scores)}) for calculating ATC accuracy using a sliding window of {window_size}"
            )

    # Filter out zero score values.
    test_scores = _filter_out_zeros(test_scores, test_scores)
    if _are_all_values_zero(test_scores.tolist()):
        print_and_log("All scores are 0, returning 0 as the accuracy.")
        return 0

    # Now calculate the accuracy and return it.
    atc_accuracy = ATC_helper.get_ATC_acc(avg_atc_threshold, test_scores)
    print_and_log(f"Calculated ATC accuracy {atc_accuracy}")
    return atc_accuracy


def _is_list_empty(data_array: list[Any]):
    """If we have no values, raise error."""
    return len(data_array) == 0 or not any(data_array)


def _are_all_values_zero(list1: list[Any]) -> bool:
    """Checks if all values in a list, including nested values, are 0."""
    for item in list1:
        if isinstance(item, list):
            if not _are_all_values_zero(item):
                return False
        elif item != 0:
            return False

    return True
