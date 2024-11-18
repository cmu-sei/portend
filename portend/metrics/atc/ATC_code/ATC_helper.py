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

# This file was obtained from https://github.com/saurabhgarg1996/ATC_code/blob/master/ATC_helper.py . It was modified according to its license permissions, as
# described in the LICENSE file in this folder.
#
# This file was modify to comply with the coding standard in our project, so the only changes are cosmetic: type hints, formatting, etc.

import typing
from typing import Any, Tuple

import numpy as np
import numpy.typing as npt


def get_entropy(probs: npt.ArrayLike) -> npt.NDArray[Any]:
    return typing.cast(
        npt.NDArray[Any],
        np.sum(np.multiply(probs, np.log(probs + 1e-20)), axis=1),  # type: ignore
    )


def get_max_conf(probs: npt.ArrayLike) -> float:
    return float(np.max(probs, axis=-1))


def find_ATC_threshold(
    scores: npt.NDArray[Any], labels: npt.NDArray[Any]
) -> Tuple[float, float]:
    sorted_idx = np.argsort(scores)

    sorted_scores = scores[sorted_idx]
    sorted_labels = labels[sorted_idx]

    # This is making the assumption that negative labels have the value of 0.
    fp = np.sum(labels == 0)
    fn = 0.0

    min_fp_fn = np.abs(fp - fn)
    thres = 0.0
    for i in range(len(labels)):
        if sorted_labels[i] == 0:
            fp -= 1
        else:
            fn += 1

        if np.abs(fp - fn) < min_fp_fn:
            min_fp_fn = np.abs(fp - fn)
            thres = sorted_scores[i]

    return min_fp_fn, thres


def get_ATC_acc(thres: float, scores: npt.NDArray[np.float_]) -> float:
    return float(np.mean(scores >= thres) * 100.0)
