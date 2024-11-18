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

from portend.experiments import experiments
from portend.monitor import alerts
from portend.monitor import config as monitor_config
from portend.utils import setup
from portend.utils.logging import print_and_log

LOG_FILE_NAME = "selector.log"


def add_selector_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Adds Selector-specific arguments and processes them."""
    parser.add_argument(
        "--expfolder",
        type=str,
        help="existing folder packaged subfolders from experiment result",
    )
    return parser


# Main code.
def main():
    # Setup logging, command line params, and load config.
    arg_parser = setup.setup_logs_and_args(LOG_FILE_NAME)
    arg_parser = add_selector_args(arg_parser)
    args, config = setup.load_args_and_config(arg_parser)
    print_and_log(f"Executing Selector, with args: {args}")

    # Step 1: Load data from predictor results (assumption: using packaged experiment result from exp_run bash)
    if "expfolder" not in args:
        raise Exception("Base folder for packaged folders was not configured.")
    (
        field,
        metric_results,
        metric_configs,
    ) = experiments.load_packaged_results(args.expfolder, config)

    # Step 2: Generate suggested alert thresholds.
    num_alert_levels = config.get("num_alert_levels")
    metric_stats = experiments.get_metric_results_stats(metric_results)
    print_and_log(f"Metric stats: {metric_stats}")
    metric_levels = alerts.calculate_metric_thresholds(
        metric_configs, metric_stats, num_alert_levels
    )
    print_and_log(f"Generated metric levels: {metric_levels}")

    # Step 3: Create and save monitor configuration file.
    monitor_config.create_monitor_config(
        metric_results,
        metric_levels,
        metric_configs,
        monitor_config_output_path=config.get("monitor_config_output_file"),
    )


if __name__ == "__main__":
    main()
