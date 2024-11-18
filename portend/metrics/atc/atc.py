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

from typing import Any, Callable, Optional

import numpy.typing as npt

from portend.analysis.predictions import Predictions
from portend.datasets.dataset import DataSet
from portend.metrics.atc import atc_functions
from portend.metrics.basic import BasicMetric
from portend.metrics.metric import MetricResult
from portend.utils.logging import print_and_log


class ATCMetric(BasicMetric):
    """Class that implements the ATC accuracy metric."""

    AVG_ATC_THRESHOLD_KEY = "average_atc_threshold"
    """Key for additional data in results."""

    SLIDING_WINDOW_KEY = "window_size"
    """Key for sliding window size in configuration."""

    DEFAULT_SLIDING_WINDOW = 7
    """Default value for sliding window."""

    prep_function: Optional[  # type: ignore
        Callable[
            [Optional[npt.NDArray[Any]], Predictions, dict[str, Any]],
            tuple[
                list[Any],
                npt.NDArray[Any],
                npt.NDArray[Any],
            ],
        ]
    ]
    """
    Optional external prep data function to be called before executing the metric, defined here to define types needed for this metric.
    It receives a list of ids for the samples, the results of the model, and a configuration dict.
    It should return 3 dicts of numpy arrays: the confidence/probability for the given dataset/predictions, the labels/truth values
    for each sample in the source dataset, and the predictions based on distance error.
    """

    result: MetricResult[dict[str, float]] = MetricResult()
    """To store the result."""

    def __init__(
        self,
        predictions: list[Predictions],
        datasets: Optional[list[DataSet]] = None,
        config: dict[str, Any] = {},
    ):
        """Constructor, should be followed by all derived classes, must call base class constructor."""
        super().__init__(predictions, datasets, config)

        self.accumulated_scores: list[float] = []
        """Sliding window of scores, if enabled will store a number of previous scores for a sliding window."""

        print_and_log(
            f"ATC metric created with config params: {self.config_params}"
        )

    def _calculate_metric(self) -> MetricResult[dict[str, float]]:
        """
        Metric implementation, will work slightly differently depending on whether one or two predictions are recieved. If two, then one is used
        to calculate reference ATC threshold, and accuracies are calculated on the second one. If one, both threshold and accuracies are calculated
        on the same set. Optionally, an ATC threshold can be configured; if that is the case, it will be used instead of calculating it in both cases.
        """
        self.result = MetricResult()
        accuracy: float = 0
        if len(self.predictions) == 1 or len(self.predictions) == 2:
            accuracy = self._calculate_atc_accuracy()
        else:
            raise RuntimeError(
                f"One or two sets of predictions are needed, but {len(self.predictions)} were received."
            )

        self.result.value = {MetricResult.METRIC_RESULTS_OVERALL_KEY: accuracy}
        return self.result

    def _calculate_atc_accuracy(self) -> float:
        """Calculate the ATC accuracy on the available data sets."""
        # First get data for the reference set: probabilities, labels, and predictions,
        # then use that to calculate the ATC threshold.
        ref_dataset = self.predictions[0]
        ref_probs, ref_labels, ref_preds = self._get_detailed_data(ref_dataset)
        atc_threshold = self._get_threshold(ref_probs, ref_labels, ref_preds)

        # Decide which target set to use, depending on whether we have a second one.
        if len(self.predictions) == 2:
            targ_probs, _, _ = self._get_detailed_data(self.predictions[1])
        else:
            targ_probs = ref_probs

        # Check if we are using a sliding window.
        window_size: int = 0
        if self.SLIDING_WINDOW_KEY in self.config_params:
            window_size = int(
                self.config_params.get(self.SLIDING_WINDOW_KEY, 0)
            )
            print_and_log(f"Found window size config in params: {window_size}")
        else:
            print_and_log("No window size config found in params")

        # Store the window size we are using, and if we are not, recommend the default one for operations.
        self.result.add_data(
            self.SLIDING_WINDOW_KEY,
            window_size if window_size != 0 else self.DEFAULT_SLIDING_WINDOW,
        )

        # Calculate the accuracy for the target data.
        accuracy = atc_functions.calculate_atc_accuracy(
            targ_probs,
            atc_threshold,
            window_size=window_size,
            accumulated_scores=self.accumulated_scores,
        )
        return accuracy

    def _get_detailed_data(
        self, predictions: Predictions
    ) -> tuple[list[Any], npt.NDArray[Any], npt.NDArray[Any]]:
        """
        Returns the proper probabilities and ATC threshold depending on the input data received.
        :return: A numpy array with a list of probabilities per sample, a numpy array of the label per sample, and a numpy array with predictions.
        """
        if self.prep_function is None:
            raise Exception(
                "Using ATC metric without a prep_function is not currently supported."
            )

        # Call external function to get specific data for probabilities/confidences and labels.
        print_and_log(f"Calling prepare data function: {self.prep_function}")
        probabilities, labels, data_predictions = self.prep_function(
            None, predictions, self.config_params
        )
        return probabilities, labels, data_predictions

    def _get_threshold(
        self,
        probabilities: list[Any],
        labels: npt.NDArray[Any],
        predictions: npt.NDArray[Any],
    ) -> float:
        """Obtains the ATC threshold to use. If it is configured, it will get it from them; otherwise, it will calculate
        the average of the provided probs and labels."""
        atc_threshold: float
        if self.AVG_ATC_THRESHOLD_KEY in self.config_params:
            atc_threshold = float(
                self.config_params.get(self.AVG_ATC_THRESHOLD_KEY, 0)
            )
            print_and_log(
                f"Config for average ATC threshold found, loading and using it: {atc_threshold}"
            )
        else:
            atc_threshold = atc_functions.calculate_atc_threshold(
                probabilities, labels, predictions
            )
            print_and_log(f"Calculated ATC threshold: {atc_threshold}")

        # Register in the results the ATC threshold that was used.
        self.result.add_data(self.AVG_ATC_THRESHOLD_KEY, atc_threshold)

        return atc_threshold
