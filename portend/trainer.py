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
from typing import Any

import numpy as np

from portend.analysis.time_series.timeseries import TimeSeries
from portend.datasets import dataset_loader
from portend.datasets.dataset import DataSet
from portend.models import ml_model
from portend.models.ts_model import TimeSeriesModel
from portend.training.model_trainer import ModelTrainer
from portend.utils import setup
from portend.utils.logging import print_and_log

LOG_FILE_NAME = "training.log"
RANDOM_SEED = 555


def aggregate_data(
    dataset_instance: DataSet, time_interval_params: dict[str, Any]
):
    """Aggregate the dataset data into a time series."""
    time_series = TimeSeries()
    if dataset_instance.has_timestamps():
        # If the dataset comes with timestamps, aggregate using them.
        time_series.aggregate_by_timestamp(
            time_interval_params.get("starting_interval"),
            time_interval_params.get("interval_unit"),
            np.array(dataset_instance.get_model_output()),
            dataset_instance.get_timestamps(),
        )
    else:
        # If the dataset doesn't have timestamps, use the samples_per_interval config to aggregate.
        samples: str = typing.cast(
            str, time_interval_params.get("samples_per_time_interval")
        )
        if samples is None:
            raise Exception("No valid samples per interval received")

        samples_per_interval = int(samples)
        time_series.aggregate_by_number_of_samples(
            time_interval_params.get("starting_interval"),
            time_interval_params.get("interval_unit"),
            np.array(dataset_instance.get_model_output()),
            samples_per_interval,
        )
    return time_series


# Main code.
def main():
    # Setup logging, command line params, and load config.
    arg_parser = setup.setup_logs_and_args(LOG_FILE_NAME)
    _, config = setup.load_args_and_config(arg_parser)

    np.random.seed(RANDOM_SEED)

    print_and_log(
        "--------------------------------------------------------------------"
    )
    print_and_log("Starting trainer session.")

    # Loading model and dataset.
    model_instance = ml_model.load_model(config.get("model"))
    main_trainer = ModelTrainer(model_instance, config.get("hyper_parameters"))
    dataset_instance = dataset_loader.load_dataset(config.get("dataset"))

    # Run steps depending on config.
    if config.get("training") == "on":
        main_trainer.split_and_train(dataset_instance)
        main_trainer.model_instance.save_to_file(config.get("output_model"))
    if config.get("cross_validation") == "on":
        main_trainer.cross_validate(dataset_instance)
    if config.get("evaluation") == "on":
        default_evaluation_input = dataset_instance.get_model_inputs()
        default_evaluation_output = dataset_instance.get_model_output()
        main_trainer.evaluate(
            default_evaluation_input, default_evaluation_output
        )
    if config.get("time_series_training") == "on":
        time_interval_params = config.get("time_interval")
        time_series = aggregate_data(dataset_instance, time_interval_params)
        print_and_log(
            f"Finished aggregating data, number of aggregated intervals: {time_series.get_num_intervals()}"
        )

        print_and_log("Training time-series model")
        ts_model = TimeSeriesModel()
        ts_model.create_model(
            time_series.get_time_intervals(),
            time_series.get_aggregated(),
            time_interval_params.get("interval_unit"),
            config.get("ts_hyper_parameters"),
        )

        print_and_log("Finished training time-series model, saving it now.")
        ts_model.save_to_file(config.get("ts_output_model"))

    print_and_log("Finished trainer session.")
    print_and_log(
        "--------------------------------------------------------------------"
    )


if __name__ == "__main__":
    main()
