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

import copy
from typing import Any, Optional

import pytest

from portend.monitor.alerts import calculate_alert_level, clear_metrics_cache

# Base config data for tests.
BASE_CONFIG = {
    "metrics": [
        {
            "name": "ATC",
            "metric_class": "portend.metrics.atc.atc.ATCMetric",
            "params": {
                "prep_module": "portend.examples.uav.wildnav_prep",
                "additional_data": "confidences",
                "average_atc_threshold": -0.36,
                "distance_error_threshold": 5,
            },
        }
    ],
    "alerts": {
        "ATC": [
            {"less_than": 40, "alert_level": "critical"},
            {"less_than": 95, "alert_level": "warning"},
        ]
    },
}


@pytest.fixture(autouse=True)
def run_around_tests():
    clear_metrics_cache()
    yield


def check_alert(
    threshold: float = -0.02,
    window: Optional[float] = None,
    additional_data: Optional[dict[str, Any]] = None,
):
    if additional_data is None:
        additional_data = {
            "confidences": {
                "Confidence": {0: "[0.87, 0.6]", 1: "[0.37, 0.5]"},
                "Confidence Invalid": {0: "[0.0, 0.001]", 1: "[0.0, 0.001]"},
                "Matched": {0: True, 1: True},
            }
        }
    config = copy.deepcopy(BASE_CONFIG)

    # Set values.
    config["metrics"][0]["params"]["average_atc_threshold"] = threshold  # type: ignore
    if window is not None:
        config["metrics"][0]["params"]["window_size"] = window  # type: ignore

    print("Testing monitor alert level selection.")
    alert_level = calculate_alert_level(None, None, additional_data, config)
    print(f"Received alert level: {alert_level}")

    return alert_level


@pytest.mark.parametrize("threshold", [-0.1, 0.0, 1])
def test_alerts_critical(threshold: float):
    alert_level = check_alert(threshold)
    assert alert_level == "critical"


@pytest.mark.parametrize("threshold", [-0.5])
def test_alerts_warning(threshold: float):
    alert_level = check_alert(threshold)
    assert alert_level == "warning"


def test_sliding_window_not_enough():
    alert_level = check_alert(window=7)

    # Not enough data for sliding window, alert level should be none
    assert alert_level == "none"


def test_sliding_window_enough():
    additional_data = {
        "confidences": {
            "Confidence": {0: "[0.87, 0.6]"},
            "Confidence Invalid": {0: "[0.0, 0.001]"},
            "Matched": {0: True},
        }
    }

    # First time no alert since there aren't enough scores.
    alert_level = check_alert(window=2, additional_data=additional_data)
    assert alert_level == "none"

    # Second time calculate actual level.
    alert_level = check_alert(window=2, additional_data=additional_data)
    assert alert_level == "critical"


def test_no_sliding_window():
    alert_level = check_alert()

    # No sliding window, alert level should be NOT none
    assert alert_level != "none"
