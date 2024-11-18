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

from typing import Any, Optional

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

from portend.utils.logging import print_and_log


class DensityEstimator:
    """Implements most common density functions."""

    dist_range: Optional[
        npt.NDArray[np.float_]
    ] = None  # Helper array with range of potential valid values used to calculate distributions.
    distribution = ""
    config_params: dict[str, Any] = {}

    def __init__(
        self,
        config_params: dict[str, Any],
    ):
        """Gets the config params and sets up the dist range. External_module may contain external implementation
        of a density function."""
        self.config_params = config_params
        self.setup_valid_range()

    def setup_valid_range(self):
        """Compute the dist range based the configuration."""
        if self.dist_range is None:
            if (
                self.config_params.get("range_start") is None
                or self.config_params.get("range_end") is None
                or self.config_params.get("range_step") is None
            ):
                raise Exception("Range not properly configured.")
            range_start = float(self.config_params.get("range_start", 0))
            range_end = float(self.config_params.get("range_end", 0))
            range_step = float(self.config_params.get("range_step", 0))
            print_and_log(f"Range: {range_start} to {range_end}")
            self.dist_range = np.arange(range_start, range_end, range_step)

    def calculate_probability_distribution(
        self, distribution: str, data, density_params: dict[str, Any]
    ) -> npt.NDArray[Any]:
        """Calculates and returns the probability distribution for the given data."""
        # print_and_log(f"Using distribution: {distribution}")
        if distribution == "normal":
            if data is None:
                return self._calculate_normal_dist_from_params(density_params)
            else:
                return self._calculate_normal_dist(data, density_params)
        else:
            raise Exception(f"Unsupported distribution type: {distribution}.")

    def _calculate_normal_dist_from_params(
        self, density_params: dict[str, Any]
    ) -> npt.NDArray[Any]:
        """Normal dist calculation, mean and std dev from params. data is ignored."""
        mean = density_params.get("mean")
        std_dev = density_params.get("std_dev")
        if mean is None or std_dev is None:
            raise Exception(
                "Can't calculate normal distribution; one of the params is None"
            )
        # print_and_log(f"Mean: {mean}, Std Dev: {std_dev}")
        dist: npt.NDArray[Any] = norm.pdf(self.dist_range, mean, std_dev)
        return dist

    def _calculate_normal_dist(
        self, data: npt.NDArray[Any], density_params: dict[str, Any]
    ) -> npt.NDArray[Any]:
        """Normal dist calculation, using only std dev from params."""
        mean = np.mean(data)
        std_dev = density_params.get("std_dev")
        if std_dev is None:
            raise Exception(
                "Can't calculate normal distribution; standard deviation provided for data is None"
            )
        # print_and_log(f"Mean: {mean}, Std Dev: {std_dev}")
        dist: npt.NDArray[Any] = norm.pdf(self.dist_range, mean, std_dev)
        return dist
