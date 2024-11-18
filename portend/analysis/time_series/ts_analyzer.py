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
from typing import Any, Dict, Optional

import portend.metrics.metric_loader as metric_loader
from portend.analysis.predictions import ClassPredictions, Predictions
from portend.analysis.time_series.timeseries import TimeSeries
from portend.datasets.dataset import DataSet
from portend.metrics.ts_metrics import TSMetric
from portend.models.ts_model import TimeSeriesModel
from portend.utils.logging import print_and_log


def analyze_ts(
    datasets: list[DataSet],
    prediction_list: list[Predictions],
    ts_config: Optional[dict[str, Any]],
    metrics_config: Optional[list[dict[str, Any]]],
) -> dict[int, dict[str, Any]]:
    """Analyzes a timeseries data."""
    if ts_config is None:
        raise Exception("Time series configuration is missing.")

    # Note that Time Series analysis only supports 1 dataset, will ignore the rest if any.
    full_dataset = datasets[0]
    predictions = typing.cast(ClassPredictions, prediction_list[0])
    if len(datasets) > 1:
        raise Exception(
            "Time Series Analysis does not support more than 1 dataset."
        )

    # Aggregate dataset and calculate original dataset classifier accuracy by time interval.
    time_interval: dict[str, Any] = typing.cast(
        Dict[str, Any], ts_config.get("time_interval")
    )
    if time_interval is None:
        raise Exception("Time interval configuration is missing.")
    time_series = TimeSeries()
    time_series.aggregate_by_timestamp(
        time_interval.get("starting_interval"),
        time_interval.get("interval_unit"),
        predictions.get_predictions(),
        full_dataset.get_timestamps(),
    )
    accuracy = calculate_accuracy(predictions, time_series)

    # Load and run time-series model on the aggregated data.
    ts_predictions = (
        None  # timeseries.create_test_time_series(0, 1000, 1001)    # TEST
    )
    try:
        print_and_log("Time-series model loading and executing.")
        ts_model = TimeSeriesModel()
        ts_model.load_from_file(typing.cast(str, ts_config.get("ts_model")))
        ts_predictions, _ = ts_model.predict(time_series)
        print_and_log("Time-series model finished running.")
    except Exception as ex:
        print_and_log(
            f"WARNING: Could not load or run time-series model: {str(ex)}"
        )
        raise ex

    # Calculate metrics and return combined results.
    metric_results = calculate_metrics(
        time_series, ts_predictions, metrics_config
    )
    metric_results = add_accuracy(metric_results, accuracy)
    return metric_results


def calculate_accuracy(
    predictions: ClassPredictions, time_series: TimeSeries
) -> dict[int, Any]:
    """Calculates the accuracy of a classifier using time intervals defined in a time series."""
    accuracy_by_interval: dict[int, Any] = {}
    starting_sample_idx = 0
    print_and_log("Calculating accuracy by interval.")
    for interval_index, _ in enumerate(time_series.get_time_intervals()):
        # Calculate the size of this interval, and obtain a slice from the current idx of that size.
        interval_sample_size = time_series.get_num_samples(interval_index)
        predictions_slice = predictions.create_slice(
            starting_sample_idx, interval_sample_size
        )

        # Calculate the accuracy of the slice and store it.
        accuracy = None
        if predictions_slice is not None:
            accuracy = predictions_slice.calculate_accuracy()
        accuracy_by_interval[interval_index] = accuracy

        # Update the starting idx for next iteration.
        starting_sample_idx += interval_sample_size

    print_and_log("Finished calculating accuracy by interval.")
    return accuracy_by_interval


def calculate_metrics(
    time_series: TimeSeries,
    ts_predictions: TimeSeries,
    metrics: Optional[list[dict[str, Any]]],
) -> dict[int, dict[str, Any]]:
    """Calculates metrics for the given configs and dataset."""
    if metrics is None:
        print_and_log("No metrics configured.")
        return {}

    results: dict[int, dict[str, Any]] = {}
    for metric_info in metrics:
        metric_name = metric_info.get("name")
        print_and_log(f"Loading metric: {metric_name}")

        metric_type = metric_loader.load_metric_type(metric_info)
        metric: TSMetric = typing.cast(
            TSMetric,
            metric_type(
                predictions=ts_predictions,
                datasets=time_series,
                config=metric_info,
            ),
        )

        print_and_log(f"Calculating metric: {metric_name}")
        for interval_index, time_interval in enumerate(
            time_series.get_time_intervals()
        ):
            if interval_index not in results.keys():
                results[interval_index] = {}
                results[interval_index]["interval"] = time_interval.timestamp()
                results[interval_index]["metrics"] = []

            # Calculate metric.
            try:
                # print_and_log(f"Calculating metric for interval: {time_interval}")
                metric.step_setup(interval_index)
                metric_results = metric.calculate_metric()
                metric_results.metric_name = metric_name
            except Exception as ex:
                print_and_log(
                    f"WARNING: Could not prepare or calculate metric {metric_name} for interval {interval_index}: {str(ex)}"
                )
                continue

            # Accumulate results.
            results[interval_index]["metrics"].append(metric_results.to_json())

    return results


def add_accuracy(
    metric_results: dict[int, dict[str, Any]], accuracy_list: dict[int, Any]
) -> dict[int, dict[str, Any]]:
    """Merges the accuracy results into the metrics results."""
    for index, accuracy in accuracy_list.items():
        metric_results[index]["accuracy"] = accuracy
    return metric_results
