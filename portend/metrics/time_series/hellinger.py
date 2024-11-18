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

import numpy as np
import numpy.typing as npt

from portend.metrics.ts_metrics import DistanceMetric


class HellingerMetric(DistanceMetric):
    # Overriden.
    def metric_distance(self, p: npt.NDArray[Any], q: npt.NDArray[Any]):
        """Calculates Hellinger distance."""
        hellinger_dist = np.sqrt(
            np.sum(np.where(p != 0, (np.sqrt(p) - np.sqrt(q)) ** 2, 0))
        ) / np.sqrt(2)
        # print(f"Hellinger Distance: {hellinger_dist}")
        return hellinger_dist


"""
if __name__ == "__main__":
    n = 1000
    p = np.random.random(n)
    q = np.random.random(n)
    p = p / np.sum(p)
    p = q / np.sum(q)
    hellinger_dist = metric_distance(p, q)
    print(f"Length of return is {hellinger_dist.size}")
"""
