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
from typing import Any, Callable, Dict, Generic, Optional, TypeVar

from portend.utils import module as module_utils
from portend.utils.timer import Timer

DEFAULT_PREP_FUNCTION = "prep_metric_data"


class Metric:
    """Generic Metric class, will load specific metric module as required."""

    predictions: Optional[Any] = None
    """Results of predicitons of an associated model."""

    datasets: Optional[Any] = None
    """Source datasets that were used in predictions."""

    config_params: dict[str, Any] = {}
    """Dictionary of configuration parameters."""

    prep_function: Optional[Callable[[Any], Any]] = None
    """Optional function to get or pre-process proper data needed by the metric, which is specific to a certain use case."""

    def __init__(
        self,
        predictions: Optional[Any] = None,
        datasets: Optional[Any] = None,
        config: dict[str, Any] = {},
    ):
        """Sets up data and config for metric."""
        self.datasets = datasets
        self.predictions = predictions

        self.config_params = typing.cast(
            Dict[str, Any], config.get("params", {})
        )
        if "prep_module" in self.config_params:
            prep_module = module_utils.load_module(
                self.config_params.get("prep_module")
            )
            self.prep_function = getattr(
                prep_module,
                self.config_params.get("prep_function", DEFAULT_PREP_FUNCTION),
            )

    def calculate_metric(self) -> MetricResult[Any]:
        """General method to calculate a metric."""
        # Run timed metric.
        timer = Timer()
        timer.start()
        result = self._calculate_metric()
        timer.stop()

        # Add recorded time to result.
        result.timer = timer
        return result

    def _calculate_metric(self) -> MetricResult[Any]:
        """Generic method to calculate the value of this metric. Should be overriden by submetrics."""
        raise NotImplementedError(
            "Method to calculate metric value must be implemented by subclass."
        )


T = TypeVar("T")


class MetricResult(Generic[T]):
    """The result of calculating a metric."""

    METRIC_NAME_KEY = "name"
    METRIC_RESULTS_KEY = "results"
    METRIC_RESULTS_OVERALL_KEY = "overall"
    METRIC_DATA_KEY = "data"
    METRIC_TIME_KEY = "time"
    METRIC_TOTAL_MS_KEY = "total_in_ms"
    METRIC_PROCESS_MS_KEY = "process_in_ms"
    """Keys used in output with metric results."""

    metric_name: Optional[str] = None
    """The name of the metric for this result."""

    value: T
    """The main value result of the metric."""

    additional_data: dict[str, Any] = {}
    """Any additional data the metric may return."""

    timer: Timer
    """Timer to store time results."""

    def add_data(self, key: str, value: Any):
        """Adds data to the internal dict."""
        self.additional_data[key] = value

    def to_json(self) -> Dict[str, Any]:
        """Converts this result to a dict."""
        metric_dict: Dict[str, Any] = {}
        metric_dict[self.METRIC_NAME_KEY] = self.metric_name
        metric_dict[self.METRIC_RESULTS_KEY] = self.value
        metric_dict[self.METRIC_DATA_KEY] = self.additional_data
        metric_dict[self.METRIC_TIME_KEY] = {
            self.METRIC_TOTAL_MS_KEY: self.timer.elapsed,
            self.METRIC_PROCESS_MS_KEY: self.timer.proc_elapsed,
        }
        return metric_dict
