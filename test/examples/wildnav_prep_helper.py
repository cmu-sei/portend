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
from typing import Any, Optional

from numpy import typing as npt

from portend.analysis.predictions import Predictions
from portend.utils import csv


def get_wildnav_test_data(
    test_file_name: str,
) -> tuple[Optional[npt.NDArray[Any]], Predictions, dict[str, Any]]:
    # Prepare path to test data.
    path_to_current_file = os.path.realpath(__file__)
    current_directory = os.path.dirname(path_to_current_file)
    test_file_path = os.path.join(current_directory, test_file_name)

    # Prepare data to sent to main function.
    ids: Optional[npt.NDArray[Any]] = None
    csv_dict = csv.load_csv_data([test_file_path])
    print(csv_dict[test_file_name].keys())
    predictions = Predictions()
    predictions.store_additional_data(csv_dict)
    params = {"additional_data": test_file_name, "distance_error_threshold": 5}

    return ids, predictions, params
