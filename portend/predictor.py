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

import argparse
from typing import Any, Optional

from portend.analysis import analysis_io, analyzer, package_io
from portend.analysis.file_keys import PredictorConfigKeys
from portend.analysis.predictions import Predictions
from portend.datasets import dataset_loader
from portend.datasets.dataset import DataSet
from portend.models import ml_model
from portend.utils import setup
from portend.utils.config import Config
from portend.utils.logging import get_full_path, print_and_log
from portend.utils.timer import Timer

LOG_FILE_NAME = "predictor.log"


def add_predictor_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Adds Predictor-specific arguments and processes them."""
    parser.add_argument(
        "--pack",
        type=str,
        help="enables packaging results in a zip file, providing the name prefix to be used for the package",
    )
    parser.add_argument(
        "--drifts",
        nargs="*",  # 0 or more values expected => creates a list
        type=str,
        help="path to the config files used to generate the drifted dataset that the predictor is using, if any",
    )
    parser.add_argument(
        "--pfolder",
        type=str,
        help="existing package folder to load predictions from",
    )
    return parser


def get_predictions(
    model: Optional[ml_model.MLModel],
    full_dataset: DataSet,
    dataset_id: int,
    dataset_config: dict[str, Any],
    config: Config,
    args: argparse.Namespace,
) -> tuple[Predictions, bool]:
    """Gets the predictions for the dataset, either applying the model or loading them."""
    predictions_input = dataset_config.get("predictions_input")
    if predictions_input is None and args.pfolder is None and model is not None:
        print_and_log("Predicting for dataset inputs...")
        prediction = analyzer.predict(
            model,
            model_input=full_dataset.get_model_inputs(),
            model_output=full_dataset.get_model_output(),
            class_params=config.get("classification"),
        )
        print_and_log("Finished predicting")
        load_mode = False
    else:
        # If the predictions input param is configured, or we got a packaged folder argument, load predictions instead of calculating them.
        if args.pfolder is not None:
            # Packaged folder argument overrides configured predictions input param.
            predictions_input = package_io.get_dataset_predictions_file(
                args.pfolder, dataset_id, dataset_config
            )

        print_and_log(f"Loading predictions from {predictions_input}")
        prediction = analysis_io.load_predictions(
            predictions_input, class_params=config.get("classification")  # type: ignore
        )
        print_and_log("Finished loading predictions")
        load_mode = True

    return prediction, load_mode


# Main code.
def main():
    # Start timer.
    timer = Timer()
    timer.start()

    # Setup logging, command line params, and load config.
    arg_parser = setup.setup_logs_and_args(LOG_FILE_NAME)
    arg_parser = add_predictor_args(arg_parser)
    args, config = setup.load_args_and_config(arg_parser)
    print_and_log(f"Executing Predictor, with args: {args}")

    # Load mode.
    mode = config.get("mode")
    print_and_log(f"Executing in mode: {mode}")

    # Load ML model/algorithm.
    model = None
    if args.pfolder is None:
        print_and_log(
            "Loading model: " + config.get("model").get("model_class")
        )
        model = ml_model.load_model(config.get("model"))
    else:
        print_and_log(
            f"Not loading model, will load predictions from packaged folder argument: {args.pfolder}"
        )

    # If we will analyze and want to package the results, prepare the folder for that.
    packaged_folder_path = "./"
    if mode == "analyze" and args.pack:
        prefix = args.pack
        packaged_folder_base = config.get(
            PredictorConfigKeys.ANALYSIS_KEY, {}
        ).get("packaged_folder")
        packaged_folder_path = package_io.create_packaged_folder(
            prefix, config.config_filename, packaged_folder_base
        )

    # Load all datasets, predict for each of them, and store results if needed.
    datasets: list[DataSet] = []
    predictions: list[Predictions] = []
    dataset_configs: list[dict[str, str]] = config.get("datasets")
    if dataset_configs is None:
        raise Exception("No datasets are configured")
    for dataset_id, dataset_config in enumerate(dataset_configs):
        # Get dataset file from package folder, if applicable.
        dataset_file = None
        if args.pfolder is not None:
            dataset_file = package_io.get_dataset_file(
                args.pfolder, dataset_id, dataset_config
            )
            print_and_log(
                f"Using dataset file from packaged results, instead of configured one: {dataset_file}"
            )

        # Load dataset to predict on.
        print_and_log("Loading dataset...")
        full_dataset = dataset_loader.load_dataset(dataset_config, dataset_file)
        datasets.append(full_dataset)

        # Predict.
        prediction, load_mode = get_predictions(
            model, full_dataset, dataset_id, dataset_config, config, args
        )
        predictions.append(prediction)

        # If in analyze or predict mode, save the predictions to a predictions file.
        if mode == "analyze" or mode == "predict":
            # Store the predictions to a file.
            analysis_io.save_predictions(
                full_dataset,
                prediction,
                output_filename=dataset_config.get("predictions_output"),
            )
        elif mode == "label":
            analysis_io.save_updated_dataset(
                full_dataset, prediction, dataset_config.get("labelled_output")
            )
        else:
            print_and_log("Unsupported mode: " + mode)

        # If we will analyze and want to package the results, copy the dataset results to the final folder so it won't be overriten by the next one.
        if mode == "analyze" and args.pack:
            package_io.store_dataset_files(
                packaged_folder_path,
                dataset_id,
                dataset_config,
                config.get("model"),
                load_mode,
            )

    # If analyzing, also calculate metrics and store them, and package everything if requested.
    if mode == "analyze":
        # Calculate metrics and store to file.
        metric_results = analyzer.analyze(datasets, predictions, config)
        analysis_io.save_metrics(
            metric_results,
            config.get(PredictorConfigKeys.ANALYSIS_KEY).get(
                PredictorConfigKeys.METRIC_OUTPUT_KEY
            ),
        )

        # If requested, package this run's results.
        if args.pack:
            package_io.package_results(
                full_folder_path=packaged_folder_path,
                config=config,
                log_file_name=get_full_path(LOG_FILE_NAME),
                drift_config_files=args.drifts,
            )

    # Stop and show time.
    S_TO_MIN = 60
    timer.stop()
    print(
        f"Elapsed time: {timer.elapsed/Timer.S_TO_MS_FACTOR/S_TO_MIN} minutes."
    )


if __name__ == "__main__":
    main()
