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

from typing import Any, Dict, List, Optional, Union

import portend.utils.files as file_utils
from portend.metrics.metric import MetricResult
from portend.utils.config import Config

AlertLevelsList = List[Dict[str, Union[float, str]]]
"""Type used for levels list in alert config."""


def load_config(config_path: str) -> dict[str, Any]:
    """
    Loads the configuration from the given path into a dictionary.
    :param config_file: A path to a file with the configuration needed for the monitor (see calculate_alert_level for config format).
    :return: A dictionary with the configuration.
    """
    if config_path is None:
        raise Exception(
            "No configuration provided, a path needs to be received."
        )
    config = Config.load(config_path)
    return config.config_data


def create_monitor_config(
    metric_results: list[dict[str, Any]],
    metric_levels: dict[str, list[float]],
    metric_configs: list[dict[str, Any]],
    monitor_config_output_path: str,
):
    """
    Creates and saves a monitor config file based on the given metric results, levels and configurations.

    :param metric_results: A list of results for each metric.
    :param metric_levels: A list of alert levels for each metric.
    :param metric_configs: A list of configurations currently used for each metric.
    :param config: the main configuration used when running the experiments.
    """
    # Modify metric configs with new data (ie, average ATC for ATC metric)
    metric_config = _update_metric_config(metric_results, metric_configs)

    # Create alert config from levels.
    alert_config = _create_alert_config(metric_config, metric_levels)

    # Create and save actual config file.
    _create_and_save_config(
        metric_config,
        alerts_by_metric=alert_config,
        output_path=monitor_config_output_path,
    )


def _update_metric_config(
    results: list[dict[str, Any]], configs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Updates metric configs depending on metric-specific changes."""
    for metric_config in configs:
        # Assume data from first metric results is enough. TODO: check if this is valid and how to handle on generation side.
        for exp_results in results:
            for metric_results in exp_results["metrics"]:
                if (
                    MetricResult.METRIC_NAME_KEY in metric_results
                    and metric_config["name"]
                    == metric_results[MetricResult.METRIC_NAME_KEY]
                ):
                    # Add all params from data in metric results, to metric configuration, so it can be passed over to monitor.
                    if "params" not in metric_config:
                        metric_config["params"] = {}
                    metric_config["params"].update(
                        metric_results[MetricResult.METRIC_DATA_KEY]
                    )
                    break

    return configs


def _create_alert_config(
    metric_infos: list[dict[str, Any]], metric_levels: dict[str, list[float]]
) -> dict[str, AlertLevelsList]:
    """Creates the config section for alerts for all given metric, given an ascending-order list of levels for each one."""
    full_config: dict[str, list[dict[str, Union[float, str]]]] = {}
    for metric_info in metric_infos:
        metric_data = _create_alert_for_metric(
            metric_info["name"], metric_levels[metric_info["name"]]
        )
        full_config.update(metric_data)
    return full_config


def _create_alert_for_metric(
    metric_name: str, levels: list[float]
) -> dict[str, AlertLevelsList]:
    """Creates the config section for alerts for a metric, given an ascending-order list of levels."""
    # Build the list of alert levels.
    config_levels: AlertLevelsList = []
    for level_num, level in enumerate(levels):
        config_levels.append(
            {"less_than": level, "alert_level": f"level{level_num+1}"}
        )

    # Build the dict with the configuration and return it.
    config: dict[str, AlertLevelsList] = {}
    config[metric_name] = config_levels
    return config


def _create_and_save_config(
    predictor_metrics_config: list[dict[str, Any]],
    alerts_by_metric: dict[str, AlertLevelsList],
    output_path: Optional[str],
):
    """Builds the monitor config file from the given parts, and saves it to disk."""
    if output_path is None:
        raise Exception(
            "No output file path given to save monitor config file."
        )

    monitor_config: dict[str, Any] = {}
    monitor_config["metrics"] = predictor_metrics_config
    monitor_config["alerts"] = alerts_by_metric

    file_utils.save_dict_to_json_file(
        monitor_config, output_path, data_name="monitor config"
    )
