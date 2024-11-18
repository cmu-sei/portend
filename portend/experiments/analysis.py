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

import statistics
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats


def to_dataframe(
    drift_range: npt.NDArray[np.float_], metrics: list[dict[str, Any]]
) -> pd.DataFrame:
    """Returns a dataframe with the metrics info by drift range."""
    metrics_by_drift: dict[float, dict[str, Any]] = {}
    drift: float
    for index, drift in enumerate(drift_range):
        metrics_by_drift[drift] = metrics[index]
    df = pd.DataFrame(metrics_by_drift)
    return df


def test_monotonic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Spearman's correlation for monotonicity
    0 is not correlated, -1 or 1 are perfect correlations (positive or negative)
    The idea here is to evaluate for what proportion of classes the detector is
    behaving monotonically with respect to the data.

    :param df: A dataframe, metric results by drift cas (column).
    :return: An updated dataframe with an added "correlation" column.
    """
    corr = list()
    x = [float(a) for a in df.columns]
    for drift_idx in df.index:
        for metric_idx, metric in enumerate(df.loc[drift_idx]):
            y = metric["result"]
            res = stats.spearmanr(x, y)
            corr.append(res.statistic)
            df.loc[drift_idx][metric_idx]["correlation"] = corr

    return df


def calculate_thresholds_by_stddev(
    df: pd.DataFrame, num_thresholds: int = 2, smallest_drift_index: int = 0
) -> list[float]:
    """
    Calculates the recommended thresholds based on average and standard deviation.

    :param df: A dataframe, metric results by drift case (column).
    :param num_thresholds: The number of thresholds to calculate.
    :param smallest_drift_index: The index of the smallest drift in the columns, by default 0.

    :return: A list of floats, which are the recommended thresholds, in ascending order.
    """

    # Calculate the mean and standard deviation of the metric for each drift case, potentially for multiple samples.
    # TODO: This doesn't work with one sample. avg and error will contain only one value, the metric in avg, and 0 in err. The thresholds will just be the metric.
    avg: list[float] = []
    err: list[float] = []
    for drift_case_idx in df.columns:
        avg.append(statistics.mean(df[drift_case_idx]))
        err.append(statistics.stdev(df[drift_case_idx]))
    print(avg)

    # Calculate thresholds using means and standard devitation of the lowest drift case.
    # The amount of thresholds depends on the num_threshold argument, and
    # it will affect how many standard deviations are used in the calculation of each threshold.
    thresholds: list[float] = []
    for thershold_num in range(1, num_thresholds + 1):
        threshold = avg[smallest_drift_index] - (
            thershold_num * err[smallest_drift_index]
        )
        thresholds.append(threshold)

    # Invert list to return thresholds in ascending order (lowest, most critical threshold first).
    thresholds.reverse()
    return thresholds
