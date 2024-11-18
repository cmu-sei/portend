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

import numpy.typing as npt

from portend.analysis.time_series.density import DensityEstimator
from portend.analysis.time_series.timeseries import TimeSeries
from portend.metrics.metric import Metric, MetricResult


class TSMetric(Metric):
    """Implements a metric that works on time series."""

    datasets: TimeSeries
    predictions: TimeSeries

    time_interval_id: int = 0

    # Overriden.
    def __init__(
        self,
        predictions: Optional[TimeSeries] = None,
        datasets: Optional[TimeSeries] = None,
        config: dict[str, Any] = {},
    ):
        """Overriden. Source_data and Predictions are assumed to be TimeSeries."""
        super().__init__(predictions, datasets, config)

    def step_setup(self, time_interval_id: int):
        """Method to be called once in each step/iteration, if any."""
        self.time_interval_id = time_interval_id


class ErrorMetric(TSMetric):
    """Implements an error-based metric that calculates error based on output."""

    # Overriden.
    def _calculate_metric(self) -> MetricResult[Any]:
        """Implements an error-based metric."""
        result: MetricResult[Any] = MetricResult()
        result.value = self.metric_error(
            self.time_interval_id, self.datasets, self.predictions
        )
        return result

    def metric_error(
        self,
        time_interval_id: int,
        time_series: TimeSeries,
        ts_predictions: TimeSeries,
    ) -> Any:
        raise NotImplementedError(
            "Calculation for error metric must be implemented by submetric."
        )


class DistanceMetric(TSMetric):
    """Implements a distance-based metric that can load metric-specific functions from a config."""

    prev_probability_distribution: npt.NDArray[Any]  # P
    curr_probability_distribution: npt.NDArray[Any]  # Q
    density_estimator: Optional[DensityEstimator] = None

    # Overriden.
    def __init__(
        self,
        predictions: Optional[TimeSeries] = None,
        datasets: Optional[TimeSeries] = None,
        config: dict[str, Any] = {},
    ):
        """Overriden. Source_data and Predictions are assumed to be TimeSeries."""
        super().__init__(predictions, datasets, config)
        self.density_estimator = DensityEstimator(self.config_params)

    # Overriden.
    def step_setup(self, time_interval_id: int):
        """Overriden."""
        super().step_setup(time_interval_id)

        if self.density_estimator is None:
            raise Exception("Density estimator has not been set up.")

        # Calculate the PD for the data given only some params, and then the PD based only on predictions.
        self.prev_probability_distribution = (
            self.density_estimator.calculate_probability_distribution(
                self.config_params.get("distribution", ""),
                self.datasets.get_aggregated(),
                self.predictions.get_pdf_params(self.time_interval_id),
            )
        )
        self.curr_probability_distribution = (
            self.density_estimator.calculate_probability_distribution(
                self.config_params.get("distribution", ""),
                None,
                self.predictions.get_pdf_params(self.time_interval_id),
            )
        )

    # Overriden.
    def _calculate_metric(self) -> MetricResult[Any]:
        """Calculates the distance defined for the current prob dist and the reference one."""
        result: MetricResult[Any] = MetricResult()
        result.value = self.metric_distance(
            self.prev_probability_distribution,
            self.curr_probability_distribution,
        )
        return result

    def metric_distance(self, p: npt.NDArray[Any], q: npt.NDArray[Any]) -> Any:
        raise NotImplementedError(
            "Calculation for error metric must be implemented by submetric."
        )
