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

import types
from typing import Any

from portend.datasets import dataset_loader
from portend.datasets.dataset import DataSet
from portend.utils import files as file_utils
from portend.utils import module as module_utils
from portend.utils import setup
from portend.utils.logging import print_and_log

LOG_FILE_NAME = "drifter.log"


def load_drift_module(
    drift_config: dict[str, Any]
) -> tuple[types.ModuleType, dict[str, Any]]:
    """Loads the drift module and params from the drift configuration."""
    print_and_log(f"Drift condition: { drift_config.get('condition') }")
    print_and_log(f"Drift module: {drift_config.get('module')}")

    params = drift_config.get("params")
    if params is None:
        params = {}
    print_and_log(f"Drift  params: {params}")

    # Import module dynamically.
    drift_module = module_utils.load_module(drift_config.get("module"))
    return drift_module, params


def main():
    # Setup logging, command line params, and load config.
    arg_parser = setup.setup_logs_and_args(LOG_FILE_NAME)
    args, config = setup.load_args_and_config(arg_parser)
    print_and_log(f"Executing Drifter, with args: {args}")

    # Load scenario data.
    drift_module, params = load_drift_module(config.get("drift_scenario"))

    # Load dataset.
    input_dataset = dataset_loader.load_dataset(config.get("dataset"))

    if args.test:
        # Run whatever drif the module implements.
        drift_module.test_drift(input_dataset, config)
    else:
        # Apply drift, and save it to regular file, and timestamped backup file.
        drifted_dataset: DataSet = drift_module.apply_drift(
            input_dataset, params
        )
        output_file = config.get("output").get("output_dataset_file")
        drifted_dataset.save_to_file(output_file)
        if "save_backup" in config.get("output"):
            file_utils.save_timestamped_backup(output_file, "drift")


if __name__ == "__main__":
    main()
