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

from portend.monitor.alerts import calculate_alert_level
from portend.utils import setup
from portend.utils.logging import print_and_log

LOG_FILE_NAME = "monitor.log"


# Main code.
def main():
    # Setup logging, command line params, and load config.
    arg_parser = setup.setup_logs_and_args(LOG_FILE_NAME)
    _, config = setup.load_args_and_config(arg_parser)

    # Load sample, prediction results from external ops program.
    # TODO: load from json from folder?
    print_and_log("Loading sample and predictions from external file.")
    sample = {}
    result = []
    additional_data = {}

    # Calculate current alert level.
    print_and_log("Calculating current alert level.")
    alert_level = calculate_alert_level(
        sample, result, additional_data, config.config_data
    )
    print_and_log(f"Action selected: {alert_level}")

    # Output the alert_level.
    # TODO: print the alert level to a JSON file?


if __name__ == "__main__":
    main()
