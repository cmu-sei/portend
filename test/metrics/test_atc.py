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

from test.examples import wildnav_prep_helper
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from portend.analysis.predictions import Predictions
from portend.examples.uav import wildnav_prep
from portend.metrics.atc import atc_functions
from portend.metrics.atc.atc import ATCMetric
from portend.metrics.metric import MetricResult

NOLA_FILE = "nu_calculated_coordinates.csv"
FIESTA_FILE = "fi_calculated_coordinates.csv"


def load_wildnav_data(
    test_file_name: str,
) -> tuple[list[list[float]], npt.NDArray[Any], npt.NDArray[Any]]:
    """Helper function to get test data from wildnav CSV output file."""
    # Load test data.
    ids, predictions, params = wildnav_prep_helper.get_wildnav_test_data(
        test_file_name
    )

    # Run wildnav prep function.
    probs, labels, preds = wildnav_prep.prep_metric_data(
        ids, predictions, params
    )
    return probs, labels, preds


def test_atc_threshold_fiesta_data():
    test_file_name = FIESTA_FILE
    probs, labels, preds = load_wildnav_data(test_file_name)
    expected_threshold = -14.738390064115718

    threshold = atc_functions.calculate_atc_threshold(probs, labels, preds)

    assert threshold == pytest.approx(expected_threshold)


def test_atc_threshold_nola_data():
    test_file_name = NOLA_FILE
    probs, labels, preds = load_wildnav_data(test_file_name)
    expected_threshold = -13.465560923254275

    threshold = atc_functions.calculate_atc_threshold(probs, labels, preds)

    assert threshold == pytest.approx(expected_threshold)


def test_atc_scores_nola_data():
    test_file_name = NOLA_FILE
    probs, _, _ = load_wildnav_data(test_file_name)

    # fmt: off
    expected_scores = [-16.994230444667547, -13.657348634817213, -14.550353706860792, -13.510831848125664, 0.0, 0.0, -14.554315731365948,
                       -9.715746238096179, -15.233818078721631, -17.301902631254627, 0.0, 0.0, -14.033378424653119, -16.521823929275115,
                       -16.007445493150858, -8.547372968018712, -16.176906313263576, 0.0, 0.0, 0.0, -7.7926475502269605, -7.10584978975803,
                       -4.082675855570478, -6.310342007484483, -11.570276672943821, -6.3000571889534065, -8.224676542567272, -4.600739792418968,
                       -7.960742336784414, -17.43382702952095, 0.0, 0.0, 0.0, -9.226406236587547, -10.82544597986168, -3.6977259961613034, -4.166700343791143,
                       0.0, -8.753903215110409, -10.919550140426878, -10.339667430622011, -13.094244379472656, -16.077663268251204,
                       -12.543223535695232, -14.069104994070038, 0.0, -16.24660520191115, -13.275101217097887, -16.650864602515487, -13.465560923254275,
                       0.0, -6.296306258976017, -8.85060372019935, -5.634163732424292, -11.253432297615044, -11.781373334210995, -4.701007751720438, 0.0,
                       -16.49166901137793, 0.0, -15.936201550922107, -15.309373007096765, 0.0, -10.085817496096888, -4.16164178859012, -3.2309595612983446,
                       -3.2043553310963744, -6.478155850146165, -5.632315732886369, -14.905786431491, -13.066467883179065, 0.0, -12.388760329141286, -15.953952714173946,
                       0.0, -14.429360224937271, 0.0, 0.0, -3.996207314742597, -2.970287815300422, -7.772966745664094, -15.474959206413454, 0.0, -15.853897900809153,
                       -15.867095299702925, 0.0, -13.609991181441082, 0.0, 0.0, 0.0, 0.0, -11.096841462761034, -15.54938778655664, -13.483339371708379, -17.09116524760047,
                       -14.527224367222408, -11.87002994779384, -16.319560425818015, -16.217987279811236, -8.973045921573403, -14.998220402276754, -15.14058913420819, 0.0,
                       -8.574171006502551, -14.146540964554086, -13.781469054734407, -15.396050972147064, -13.250271303680666, -2.597983768291356, -4.613388451598594,
                       -15.9642382283299, -11.926491778779766, -10.607367326676433, -14.361956382424813, -11.750210519050974, -16.55545345236375, -7.6651348309787855,
                       -16.06677776299082, -3.094374557770727, -7.575760165818626, -15.1948576346295, -3.4719183637123368, -7.984370288253538, -3.9980955465915584,
                       -12.401569869001301, -10.846967116679744, -7.412872074422217, -14.609263309812466, -17.328665537211176, -3.7918247354200716, -16.231189558333917,
                       -9.537276944486493, -5.0981434542485475, -12.805002968131356, -14.421430968319521, -3.305473432095847, -6.864366069938124, -15.831206455402837,
                       -10.772250846323711, -17.2149163698395, 0.0, -17.260296206188823, -7.042487553014448, -14.969822563949108, -6.994626471718412, -14.049881578724873,
                       -9.988360105694719, -14.174141569189118, -7.313028077102278, -13.610686477215589, -13.025694362735447, -12.669789031439928, -17.33108325010559,
                       -16.219616794000032, -17.601489914958737, -17.596347929529042, -6.22137662170671, -7.6576176033823495, -6.374400048256348, -17.066682135963923,
                       -7.91380390971887, -3.1690368195853447, -7.805430588856316, -16.185693797425724, -3.6922941039521375, -8.929933086402517, -17.526746943481477,
                       -15.332396113601625, -10.830810357549037]
    # fmt: on

    scores = atc_functions._calculate_atc_scores(probs)
    print(scores.tolist())

    assert np.allclose(scores, np.array(expected_scores))


def test_accuracy_nola_data():
    test_file_name = NOLA_FILE
    probs, _, _ = load_wildnav_data(test_file_name)
    threshold = -7.637565929368777
    expected_acc = 24.822695035460992

    accuracy = atc_functions.calculate_atc_accuracy(probs, threshold)

    assert accuracy == pytest.approx(expected_acc, rel=0.001)


def test_score_empty_probs_exception() -> None:
    """Checks that empty probs for scores raises error."""
    probs: list[list[float]] = [[]]

    with pytest.raises(RuntimeError):
        _ = atc_functions._calculate_atc_scores(probs)


def test_acc_empty_probs_exception() -> None:
    """Checks that empty probs for accuracy raises error."""
    threshold = -7
    probs: list[list[float]] = [[]]

    with pytest.raises(RuntimeError):
        _ = atc_functions.calculate_atc_accuracy(probs, threshold)


def test_acc_zero_probs_exception() -> None:
    """Checks that empty probs for accuracy raises error."""
    threshold = -7
    probs = [[0, 0], [0, 0]]

    with pytest.raises(RuntimeError):
        _ = atc_functions.calculate_atc_accuracy(probs, threshold)


@pytest.mark.parametrize("threshold", [-0.1, 0.0, 1])
def test_atc_accuracy_zero(threshold: float) -> None:
    probs = [[0.04578296, 0.01576773, 0, 0]]
    expected_accuracy = 0

    accuracy = atc_functions.calculate_atc_accuracy(probs, threshold)
    print(f"Accuracy: {accuracy}")

    assert accuracy == expected_accuracy


def test_atc_accuracy_fifty() -> None:
    threshold = -0.7
    probs = [[0.5, 0.3], [0.6, 0.4]]
    expected_accuracy = 50

    accuracy = atc_functions.calculate_atc_accuracy(probs, threshold)
    print(f"Accuracy: {accuracy}")

    assert accuracy == expected_accuracy


def test_atc_accuracy_window() -> None:
    threshold = -14.7
    window_size = 7
    expected_accuracy = 57.14285714285714

    # fmt: off
    prev_scores = [-13.14762338, -6.70919908, -9.36763549, -15.46571841, -13.93413885,
                   -11.54673022, -12.87839624, -10.37840669, -15.75467847, -6.47528751,
                   -4.58614885, -6.69346101, -13.58237601, -12.47516229, -8.42227124,
                   -14.73839006, -12.07624845, -6.89364139, -8.04236115, -12.68604465,
                   -13.4187098, -12.67465405, -12.55750669, -14.87356079, -12.33160238,
                   -12.8101687, -12.48683766, -13.28716487, -11.33336263, -15.80553848,
                   -15.6993558, -9.49269813, -7.96187469, -7.07452522, -15.92856094]
    # fmt: on
    probs = [0.5]

    accuracy = atc_functions.calculate_atc_accuracy(
        probs, threshold, window_size, prev_scores
    )
    print(f"Accuracy: {accuracy}")

    assert accuracy == expected_accuracy


def test_atc_accuracy_window_not_enough_scores() -> None:
    threshold = -14.7
    window_size = 7

    prev_scores = [
        -13.14762338,
        -6.70919908,
        -9.36763549,
        -15.46571841,
        -13.93413885,
    ]
    probs = [0.5]

    with pytest.raises(RuntimeError):
        _ = atc_functions.calculate_atc_accuracy(
            probs, threshold, window_size, prev_scores
        )


def test_atc_accuracy_metric() -> None:
    """Tests the general metric calculation for ATC."""
    # Prepare.
    expected_accuracy = 40
    config = {
        "params": {
            "prep_module": "portend.examples.uav.wildnav_prep",
            "additional_data": "confidences",
            ATCMetric.AVG_ATC_THRESHOLD_KEY: -0.4,
            "distance_error_threshold": 5,
        }
    }
    additional_data = {
        "confidences": {
            "Confidence": {
                0: "[0.87, 0.6]",
                1: "[0.37, 0.5]",
                2: "[0.27, 0.7]",
                3: "[0.88, 0.99]",
                4: "[0.88, 0.99]",
            },
            "Confidence Invalid": {
                0: "[0.0, 0.001]",
                1: "[0.0, 0.001]",
                2: "[0.0, 0.001]",
                3: "[0.0, 0.001]",
                4: "[0.0, 0.001]",
            },
            "Matched": {0: True, 1: True, 2: True, 3: True, 4: True},
        }
    }
    prediction = Predictions()
    prediction.store_additional_data(additional_data)
    metric = ATCMetric([prediction], config=config)

    # Calculate.
    accuracy = metric.calculate_metric()

    # Check
    assert (
        accuracy.value[MetricResult.METRIC_RESULTS_OVERALL_KEY]
        == expected_accuracy
    )
