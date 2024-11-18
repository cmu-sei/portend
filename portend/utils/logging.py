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

import logging
import os
from typing import Optional

LOG_FOLDER = "./logs"


def setup_logging(log_file_name: str):
    """Set up file for logging."""
    log_path = get_full_path(log_file_name)
    if os.path.exists(log_path):
        os.remove(log_path)
    os.makedirs(LOG_FOLDER, exist_ok=True)
    print(f"Logging to {log_path}")
    logging.basicConfig(
        filename=log_path, format="%(asctime)s %(message)s", level=logging.DEBUG
    )


def print_and_log(message: Optional[str]):
    """Print to console, as well as log to file."""
    print(message, flush=True)
    logging.info(message)


def get_full_path(log_file_name: str) -> str:
    return os.path.join(LOG_FOLDER, log_file_name)
