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
from typing import Any, Optional, Type

from portend.analysis.predictions import Predictions
from portend.datasets.dataset import DataSet
from portend.metrics import metric_loader
from portend.metrics.metric import Metric, MetricResult


class BasicMetric(Metric):
    """Implements the simplest base metric class, with base Prediction and Dataset types for its data. Metrics should derive from this class."""

    # Definition of specific subtypes used for these types of metrics.
    predictions: list[Predictions] = []
    datasets: Optional[list[DataSet]] = None

    def __init__(
        self,
        predictions: list[Predictions],
        datasets: Optional[list[DataSet]] = None,
        config: dict[str, Any] = {},
    ):
        """Constructor, should be followed by all derived classes, must call base class constructor."""
        super().__init__(predictions, datasets, config)

    @staticmethod
    def create_from_info(
        metric_info: dict[str, Any],
        predictions: list[Predictions],
        datasets: Optional[list[DataSet]],
        config: dict[str, Any],
    ) -> BasicMetric:
        """Dynamically creates an instance of this class from the given info."""
        metric_type = typing.cast(
            Type[BasicMetric], metric_loader.load_metric_type(metric_info)
        )
        metric = metric_type(
            predictions=predictions, datasets=datasets, config=config
        )
        return metric

    def _calculate_metric(self) -> MetricResult[dict[str, float]]:
        """Generic method to calculate the value of this metric. Should be overriden by submetrics."""
        raise NotImplementedError(
            "Method to calculate metric value must be implemented by subclass."
        )
