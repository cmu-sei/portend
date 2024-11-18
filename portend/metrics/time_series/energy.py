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

from typing import Any

import numpy.typing as npt
from scipy.stats import energy_distance

from portend.metrics.ts_metrics import DistanceMetric


class EnergyMetric(DistanceMetric):
    # Overriden.
    def metric_distance(self, p: npt.NDArray[Any], q: npt.NDArray[Any]):
        """Calculates Energy distance."""
        energy_dist = energy_distance(p, q)
        # print(f"Energy distance: {energy_dist}")
        return energy_dist


"""
if __name__ == "__main__":
    n = 1000
    p = np.random.random(n)
    q = np.random.random(n)
    p = p / np.sum(p)
    p = q / np.sum(q)
    energy_dist = metric_distance(p, q)
    print(f"Length of return is {np.array(energy_dist).size}")
"""
