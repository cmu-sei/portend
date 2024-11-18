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

from portend.analysis.file_keys import PredictorConfigKeys
from portend.analysis.predictions import Predictions, build_predictions_object
from portend.analysis.time_series.ts_analyzer import analyze_ts
from portend.datasets.dataset import DataSet
from portend.metrics.basic import BasicMetric
from portend.models.ml_model import MLModel
from portend.utils.config import Config
from portend.utils.logging import print_and_log
from portend.utils.typing import SequenceLike


def predict(
    model: MLModel,
    model_input: list[SequenceLike],
    model_output: SequenceLike,
    class_params: Optional[dict[str, Any]],
) -> Predictions:
    """Generates predictions based on model, returns object with predictions."""
    raw_predictions, additional_data = model.predict(model_input)
    return build_predictions_object(
        raw_predictions,
        expected_output=model_output,
        additional_data=additional_data,
        class_params=class_params,
    )


def analyze(
    datasets: list[DataSet], predictions: list[Predictions], config: Config
) -> dict[Any, Any]:
    """Analyzes the data."""
    if config.contains(PredictorConfigKeys.TIME_SERIES_KEY):
        return analyze_ts(
            datasets,
            predictions,
            config.get(PredictorConfigKeys.TIME_SERIES_KEY),
            config.get(PredictorConfigKeys.ANALYSIS_KEY).get(
                PredictorConfigKeys.METRICS_KEY
            ),
        )
    else:
        return calculate_metrics(
            datasets,
            predictions,
            config.get(PredictorConfigKeys.ANALYSIS_KEY).get(
                PredictorConfigKeys.METRICS_KEY
            ),
        )


def calculate_metrics(
    datasets: Optional[list[DataSet]],
    predictions: list[Predictions],
    metrics: Optional[list[dict[str, Any]]],
    metric_cache: Optional[dict[str, BasicMetric]] = None,
) -> dict[str, list[dict[str, Any]]]:
    """
    Calculates metrics for the given configs and dataset, assuming BasicMetrics or derived ones.
    # param metric_cache: a dict of metrics, useful if we want to maintain some state of a metric between runs.

    :return: a dictionary with a "metrics" key pointing to a list of metric results. Each result will be a JSON dict of the metric result.
    """
    if metrics is None:
        print_and_log("No metrics configured.")
        return {}

    results: dict[str, list[dict[str, Any]]] = {}
    for metric_info in metrics:
        metric_name = metric_info.get("name")
        if metric_name is None:
            raise Exception("Invalid metric found, must have a name")

        metric: BasicMetric
        if metric_cache is not None and metric_name in metric_cache:
            # If using cache of metric objects, get metric from it.
            print_and_log(f"Retrieving metric object from cache: {metric_name}")
            metric = metric_cache[metric_name]

            # Updating predictions and datasets, to ensure new data is being used.
            metric.datasets = datasets
            metric.predictions = predictions
        else:
            print_and_log(f"Loading metric: {metric_name}")
            metric = BasicMetric.create_from_info(
                metric_info=metric_info,
                predictions=predictions,
                datasets=datasets,
                config=metric_info,
            )

            # Update cache of metric objects, if it is being used.
            if metric_cache is not None:
                metric_cache[metric_name] = metric

        print_and_log(f"Calculating metric: {metric_name}")
        results["metrics"] = []

        # Calculate metric.
        try:
            metric_result = metric.calculate_metric()
            metric_result.metric_name = metric_name
        except Exception as ex:
            print_and_log(
                f"WARNING: Could not prepare or calculate metric {metric_name}: {type(ex).__name__} - {str(ex)}"
            )
            # import traceback
            # print(traceback.format_exc())
            continue

        # Accumulate results.
        results["metrics"].append(metric_result.to_json())

    return results
