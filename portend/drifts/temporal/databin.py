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

import random
from typing import Any, Optional

import numpy.typing as npt

from portend.utils.typing import SequenceLike


def create_bins(bin_info_list: list[list[Any]], shuffle: bool) -> list[DataBin]:
    """Creates a list of bins based on a string list of bin info."""
    bins: list[DataBin] = []
    for bin_info in bin_info_list:
        bins.append(DataBin(bin_info[0], bin_info[1], shuffle))
    return bins


def sort_into_bins(
    ids: npt.NDArray[Any], values: SequenceLike, bins: list[DataBin]
):
    """Sorts the ids into bins, based on the given values matching bin values."""
    for sample_idx, value in enumerate(values):
        for bin_idx, bin in enumerate(bins):
            if value == bin.value:
                bins[bin_idx].add(ids[sample_idx])
                break

    return bins


class DataBin:
    """Class to represent a bin of data of a certain classification."""

    def __init__(self, new_name: str, new_value: Any, shuffle=True):
        self.name = new_name
        self.value = new_value
        self.shuffle = shuffle
        self.ids: list[Any] = []
        self.id_queue: list[Any] = []

    def setup_queue(self):
        self.id_queue = self.ids.copy()
        if self.shuffle:
            random.shuffle(self.id_queue)

    def get_queue_length(self) -> int:
        return len(self.id_queue)

    def pop_from_queue(self) -> Optional[Any]:
        if len(self.id_queue) > 0:
            return self.id_queue.pop(0)
        else:
            return None

    def add(self, new_id: Any):
        self.ids.append(new_id)

    def get_random(self) -> Any:
        return random.choice(self.ids)

    def size(self) -> int:
        return len(self.ids)

    def info(self) -> str:
        return self.name + ": " + str(self.size()) + " samples "

    def to_string(self) -> str:
        return self.name + ": " + str(self.ids)
