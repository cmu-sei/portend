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

from portend.datasets import image_dataset
from portend.experiments import scenario
from portend.utils import dataframe_helper, setup
from portend.utils.logging import print_and_log

LOG_FILE_NAME = "helper.log"


def add_helper_gen_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Adds Helper-specific arguments and processes them."""
    parser.add_argument(
        "-o",
        type=str,
        help="output folder to generate configurations in",
    )
    return parser


# Main code.
def main():
    # Setup logging, command line params, and load config.
    arg_parser = setup.setup_logs_and_args(LOG_FILE_NAME)
    arg_parser = add_helper_gen_args(arg_parser)
    args, config = setup.load_args_and_config(arg_parser)

    action = config.get("action")
    if action == "merge":
        print_and_log("Running merge action.")
        dataframe_helper.merge_files(
            config.get("dataset1"), config.get("dataset2"), config.get("output")
        )
    elif action == "image_json":
        print_and_log(
            "Running JSON dataset generation for image datasets action."
        )
        image_dataset.create_image_dataset_to_file(
            config.get("dataset_class"),
            config.get("image_folder"),
            config.get("extensions"),
            config.get("fields"),
            config.get("existing_data"),
            config.get("json_file"),
        )
    elif action == "config_gen":
        print_and_log("Running config generation action.")
        scenario.generate_config_files(
            config.get("base_file"),
            output_folder=args.o,
            fields=config.get(scenario.DEFAULT_FIELDS_KEY),
        )
    else:
        raise Exception(f"Unsupported action: {action}")


if __name__ == "__main__":
    main()
