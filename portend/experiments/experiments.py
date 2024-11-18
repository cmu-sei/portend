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

import os
import typing
from pathlib import Path
from typing import Any, Dict, List

from portend.analysis import analysis_io, package_io
from portend.analysis.file_keys import PredictorConfigKeys
from portend.experiments import scenario
from portend.experiments.scenario import ScenarioField
from portend.metrics.metric import MetricResult
from portend.utils.config import Config
from portend.utils.logging import print_and_log

DEFAULT_GEN_CONFIG_PREFIX = "01_helper"


def load_packaged_results(
    packaged_folder_base: str,
    config: Config,
) -> tuple[ScenarioField, list[dict[str, Any]], list[dict[str, Any]],]:
    """
    Loads information about experiment results that were stored in packaged folders.

    :param packaged_folder_base: Path to folder with one or more subfolders from a packaged experiment.
    :param config: A configuration file with at least the "generator_config_file_prefix" param, with the prefix of the generator file.

    :return: A Scenario field with the description of the field being used to generate drift, a list of metric results for each of those drift values,
             and a dictionary of the configurations used for each metric.
    """
    # Load packaged folder list.
    packaged_folders = load_experiment_folders(packaged_folder_base)

    # Get exp drift gen config file data.
    scenario_field = load_drift_scenario(
        packaged_folders,
        config.get("generator_config_file_prefix", DEFAULT_GEN_CONFIG_PREFIX),
    )
    print_and_log(f"Drift range: {scenario_field.range}")

    # Load predictor config data.
    analysis_config = load_analysis_config(packaged_folders)
    metrics_config: list[dict[str, Any]] = typing.cast(
        List[Dict[str, Any]],
        analysis_config.get(PredictorConfigKeys.METRICS_KEY),
    )

    # Load metric file data.
    metric_filename = Path(
        str(analysis_config.get(PredictorConfigKeys.METRIC_OUTPUT_KEY))
    ).name
    metrics = load_metric_results(packaged_folders, metric_filename)

    # Check consistency between scenarios and metric data.
    if len(metrics) != len(scenario_field.range):
        raise Exception(
            "Not enough metric files found for full range of experiments"
        )

    # Return results.
    print_and_log(
        f"Experiment info - field: <{scenario_field}>, metrics: <{metrics}>, metric config: <{metrics_config}>"
    )
    return scenario_field, metrics, metrics_config


def load_experiment_folders(packaged_folder_base: str) -> list[str]:
    """Loads a list of folders inside the given base folder. Assumes this is a list of packaged folders from an experiment (multiple runs of Predictor)."""
    print_and_log(f"Base folder for exp folders: {packaged_folder_base}")
    packaged_folders = [
        f.path for f in os.scandir(packaged_folder_base) if f.is_dir()
    ]
    if len(packaged_folders) == 0:
        raise Exception(f"No packaged folders found in {packaged_folder_base}")
    print_and_log(f"Experiment folders found: {packaged_folders}")
    return packaged_folders


def load_drift_scenario(
    exp_folders: list[str], gen_conf_prefix: str = DEFAULT_GEN_CONFIG_PREFIX
) -> ScenarioField:
    """Loads info about field used for drift generation, from one of the experiment packaged folders provided."""
    # Drift config file can be obtained from any folder, all of them should have the same copy of the gen file since it is common to the whole experiment.
    exp_folder_path = exp_folders[0]
    drift_folder_path = os.path.join(
        exp_folder_path, package_io.DRIFT_CONFIG_FOLDER
    )

    # Find the file.
    drift_gen_conf_file = None
    for file in os.scandir(drift_folder_path):
        if file.is_file() and file.name.startswith(gen_conf_prefix):
            drift_gen_conf_file = file
            break
    if drift_gen_conf_file is None:
        raise Exception("Drift generation file not found in packaged folders.")

    # Load the scenario field data from the file.
    return scenario.load_scenario_field(drift_gen_conf_file.path)


def load_metric_results(
    exp_folders: list[str], metric_filename: str
) -> list[dict[str, Any]]:
    """Load metric data for all metric files in the exp folders."""
    metric_results = []
    for packaged_folder in exp_folders:
        metrics_path = os.path.join(packaged_folder, metric_filename)
        metric_results.append(analysis_io.load_metrics(metrics_path))
    print_and_log(f"Found {len(metric_results)} metric files.")
    return metric_results


def load_analysis_config(exp_folders: list[str]) -> dict[str, Any]:
    """
    Loads the config section containing analysis and metric config from a predictor configuration in the experiment folders.

    :param exp_folders: The list of packaged folders with the results for each run of the experiment.

    :return: A dict with the analysis configuration, including metric information used in the experiment.
    """
    # Predictor config file can be obtained from any folder, all of them should have the same copy of the predictor file since it is common to the whole experiment.
    exp_folder_path = exp_folders[0]

    # First find the predictor config file as the only json file inside the default config folder for this experiment.
    predictor_config_file = None
    predictor_config_folder = os.path.join(
        exp_folder_path, package_io.PREDICTOR_CONFIG_FOLDER
    )
    for file in os.scandir(predictor_config_folder):
        if file.is_file() and file.name.endswith("json"):
            predictor_config_file = file
            break
    if predictor_config_file is None:
        raise Exception("Predictor config file not found in packaged folders.")
    predictor_config = Config.load(predictor_config_file.path)

    # Now get the metric config section from the file, and return it.
    if not predictor_config.contains(PredictorConfigKeys.ANALYSIS_KEY):
        raise Exception(
            "Can't load metrics config data, no analysis section found in predictor config file"
        )
    if PredictorConfigKeys.METRICS_KEY not in predictor_config.get(
        PredictorConfigKeys.ANALYSIS_KEY
    ):
        raise Exception(
            "Can't load metrics config data, no metrics section found in predictor config file"
        )
    predictor_metrics_config: dict[str, Any] = predictor_config.get(
        PredictorConfigKeys.ANALYSIS_KEY
    )
    return predictor_metrics_config


def get_metric_results_stats(
    results: list[dict[str, Any]]
) -> dict[str, dict[str, float]]:
    """
    Return statistics about the list of metric results.

    :return: A dict of dicts with max, min, and num_results per metric.
    """
    all_stats: dict[str, dict[str, float]] = {}
    stats = {"max": 0.0, "min": 0.0, "num": 0.0}

    # Iterate over all experiments, updating data for each metric.
    for exp_results in results:
        for metric_results in exp_results["metrics"]:
            # Add this metric to the final dict if it is not there yet.
            metric_name = metric_results[MetricResult.METRIC_NAME_KEY]
            if metric_name not in all_stats:
                all_stats[metric_name] = stats.copy()
                all_stats[metric_name]["max"] = metric_results[
                    MetricResult.METRIC_RESULTS_KEY
                ][MetricResult.METRIC_RESULTS_OVERALL_KEY]
                all_stats[metric_name]["min"] = metric_results[
                    MetricResult.METRIC_RESULTS_KEY
                ][MetricResult.METRIC_RESULTS_OVERALL_KEY]

            # Increase number of results for this metric.
            all_stats[metric_name]["num"] += 1

            # Track min and maxes.
            if (
                metric_results[MetricResult.METRIC_RESULTS_KEY][
                    MetricResult.METRIC_RESULTS_OVERALL_KEY
                ]
                > all_stats[metric_name]["max"]
            ):
                all_stats[metric_name]["max"] = metric_results[
                    MetricResult.METRIC_RESULTS_KEY
                ][MetricResult.METRIC_RESULTS_OVERALL_KEY]
            if (
                metric_results[MetricResult.METRIC_RESULTS_KEY][
                    MetricResult.METRIC_RESULTS_OVERALL_KEY
                ]
                < all_stats[metric_name]["min"]
            ):
                all_stats[metric_name]["min"] = metric_results[
                    MetricResult.METRIC_RESULTS_KEY
                ][MetricResult.METRIC_RESULTS_OVERALL_KEY]

    return all_stats
