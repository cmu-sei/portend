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

import datetime
import os
import shutil
from pathlib import Path
from typing import Any, Optional

from portend.analysis.file_keys import PredictorConfigKeys
from portend.analysis.predictions import Predictions
from portend.utils import files as file_utils
from portend.utils.config import Config
from portend.utils.logging import print_and_log

DEFAULT_PACKAGED_FOLDER_BASE = "output/packaged/"
PREDICTOR_CONFIG_FOLDER = "predictor_config"
DRIFT_CONFIG_FOLDER = "drift_config"
DATASET_FOLDER_PREFIX = "dataset-"


def create_packaged_folder(
    package_prefix: str,
    config_filename: str,
    packaged_folder_base: Optional[str],
) -> str:
    """Create time-stamped folder to store results. Returns the full path to this folder."""
    if packaged_folder_base is None:
        packaged_folder_base = DEFAULT_PACKAGED_FOLDER_BASE

    print_and_log(f"Base folder for stored results: {packaged_folder_base}")
    config_descriptor = os.path.splitext(os.path.basename(config_filename))[0]
    package_folder_name = (
        f"{package_prefix}-{config_descriptor}-"
        + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    )
    full_folder_path = os.path.join(packaged_folder_base, package_folder_name)
    print_and_log(f"Creating subfolder for exp results: {full_folder_path}")
    file_utils.recreate_folder(full_folder_path)
    return full_folder_path


def store_dataset_files(
    store_folder_path: str,
    dataset_id: int,
    dataset_config: dict[str, Any],
    model_config: dict[str, Any],
    load_mode: bool = False,
):
    """Stores the dataset files in their own subfolders to be packaged later."""
    file_paths = []
    if "dataset_file" in dataset_config:
        file_paths.append(str(dataset_config.get("dataset_file")))
    if "predictions_output" in dataset_config:
        predictions_filepath = str(dataset_config.get("predictions_output"))
        file_paths.append(predictions_filepath)

        # If it is there, also add the additional data file.
        pred_add_data_path = Predictions.get_add_data_filename(
            predictions_filepath
        )
        if os.path.exists(pred_add_data_path):
            file_paths.append(pred_add_data_path)

    if not load_mode:
        extra_output_files = model_config.get("model_extra_output_files")
        if extra_output_files is not None:
            for extra_file in extra_output_files:
                file_paths.append(extra_file)

    # Store the files in a subfolder of the store folder, for this specific dataset.
    dataset_store_folder_path = str(
        _get_dataset_path(store_folder_path, dataset_id)
    )
    _copy_files(dataset_store_folder_path, file_paths, folder_paths=[])


def package_results(
    full_folder_path: str,
    config: Config,
    log_file_name: str,
    drift_config_files: Optional[list[str]] = None,
):
    """Copies all results to a date-time folder to store experiment results."""
    # Create subfolder for the config used for this run of the Predictor, and copy it there to be included in the package.
    config_folder = os.path.join(full_folder_path, PREDICTOR_CONFIG_FOLDER)
    _copy_files(
        store_folder_path=config_folder, file_paths=[config.config_filename]
    )

    # Optionally, the drift config used for the dataset that was read can be attached as well.
    # TODO: support multiple drift files with same name in different paths.
    if drift_config_files is not None:
        drift_folder = os.path.join(full_folder_path, DRIFT_CONFIG_FOLDER)
        _copy_files(
            store_folder_path=drift_folder, file_paths=drift_config_files
        )

    # Add log file to list of files to copy.
    file_paths = [log_file_name]

    # Get TS model if needed.
    if config.contains(PredictorConfigKeys.TIME_SERIES_KEY):
        if "ts_model" in config.get(PredictorConfigKeys.TIME_SERIES_KEY):
            file_paths.append(
                str(
                    config.get(PredictorConfigKeys.TIME_SERIES_KEY).get(
                        "ts_model"
                    )
                )
            )

    # Add metric results to files to be stored.
    if config.contains(PredictorConfigKeys.ANALYSIS_KEY):
        if PredictorConfigKeys.METRIC_OUTPUT_KEY in config.get(
            PredictorConfigKeys.ANALYSIS_KEY
        ):
            file_paths.append(
                str(
                    config.get(PredictorConfigKeys.ANALYSIS_KEY).get(
                        PredictorConfigKeys.METRIC_OUTPUT_KEY
                    )
                )
            )

    # Check if model is file or folder, and add it to corresponding list.
    folder_paths = []
    if "model_file" in config.get("model"):
        model = config.get("model").get("model_file")
        if os.path.isfile(model):
            file_paths.append(model)
        else:
            folder_paths.append(model)

    # Copy all other files and folders to the folder to be zipped.
    _copy_files(full_folder_path, file_paths, folder_paths)

    # Package all into a zip file.
    _zip_store_folder(full_folder_path)


def _copy_files(
    store_folder_path: str, file_paths: list[str], folder_paths: list[str] = []
):
    """Copies the given files and folders to the store folder."""
    # Ensure folder exists.
    print_and_log(f"Copying files to {store_folder_path}")
    print_and_log(f"Files to copy: {file_paths}")
    os.makedirs(store_folder_path, exist_ok=True)

    # Copy all files and folders to the folder to be packaged.
    for file_path in file_paths:
        shutil.copy(file_path, store_folder_path)
    for folder_path in folder_paths:
        shutil.copytree(
            folder_path,
            os.path.join(
                store_folder_path,
                os.path.basename(os.path.normpath(folder_path)),
            ),
        )


def _zip_store_folder(full_folder_path: str):
    """Creates a zip file of the stored packaged folder."""
    print_and_log(f"Storing exp results in zip file from {full_folder_path}")
    shutil.make_archive(full_folder_path, "zip", full_folder_path)
    print_and_log("Finished backing up results.")


def get_dataset_file(
    packaged_folder_path: str, dataset_id: int, dataset_param: dict[str, Any]
):
    """Returns the path of a dataset file from an expanded packaged folder."""
    try:
        file_path = _get_dataset_packaged_file_path(
            packaged_folder_path, dataset_id, dataset_param, "dataset_file"
        )
        return file_path
    except RuntimeError:
        raise RuntimeError(
            "No dataset file was configured, can't load packaged dataset file."
        )


def get_dataset_predictions_file(
    packaged_folder_path: str, dataset_id: int, dataset_param: dict[str, Any]
):
    """Returns the path of a dataset predictions file from an expanded packaged folder."""
    try:
        file_path = _get_dataset_packaged_file_path(
            packaged_folder_path,
            dataset_id,
            dataset_param,
            "predictions_output",
        )
        return file_path
    except RuntimeError:
        raise RuntimeError(
            "No output predictions file was configured, can't load packaged predictions file."
        )


def _get_dataset_packaged_file_path(
    packaged_folder_path: str,
    dataset_id: int,
    dataset_param: dict[str, Any],
    file_key: str,
):
    """Returns the path inside of the packaged dataset folder of the configured original file path."""
    original_path = dataset_param.get(file_key)
    if original_path is None:
        raise RuntimeError(
            f"Configured file key not found in config: {original_path}"
        )
    filename = Path(original_path).name

    dataset_path = _get_dataset_path(packaged_folder_path, dataset_id)
    packaged_file_path = str(Path(dataset_path, filename))
    return packaged_file_path


def _get_dataset_path(base_path: str, dataset_id: int) -> Path:
    """Returns the path of the dataset folder for a given dataset id and base path."""
    return Path(base_path, f"{DATASET_FOLDER_PREFIX}{dataset_id}")
