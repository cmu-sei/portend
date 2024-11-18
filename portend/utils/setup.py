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
import sys

from portend.utils import logging
from portend.utils.config import Config


def setup_logs_and_args(
    log_file_name: str,
) -> argparse.ArgumentParser:
    """Sets up basic log and common args processing."""
    # Set up logging.
    logging.setup_logging(log_file_name)

    # Get parser with common args.
    return _prepare_parser()


def load_args_and_config(
    arg_parser: argparse.ArgumentParser,
) -> tuple[argparse.Namespace, Config]:
    """Parses args and loads config."""
    args = arg_parser.parse_args()
    config = Config.get_config_from_args(args)
    return args, config


def _prepare_parser() -> argparse.ArgumentParser:
    """Sets up the parser, adding common arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", help="run in test mode", action="store_true")
    parser.add_argument(
        "-c", type=str, help="config file path relative to main project folder"
    )

    if len(sys.argv) < 2:
        print("No command line arguments.")

    return parser
