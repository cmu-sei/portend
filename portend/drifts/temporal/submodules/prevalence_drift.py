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
import typing
from typing import Any, Optional

import numpy

from portend.datasets.dataset import DataSet
from portend.utils.logging import print_and_log


class TargetPrevalences:
    """Target prevalences defined in configuration."""

    target_prevalences: Optional[dict[int, Any]] = None

    def get_all_prevalences(self):
        return self.target_prevalences

    def load_prevalence_arrays(
        self, params: dict[str, Any], num_total_bins: int
    ):
        """Load and init the configured target prevalences arrays by bin, if needed."""
        if self.target_prevalences is None:
            self.target_prevalences = {}
            max_num_samples = int(params.get("max_num_samples", 1))
            sample_group_size = int(params.get("sample_group_size", 1))
            num_sample_groups = int(max_num_samples / sample_group_size)
            configured_prevalences: Optional[
                dict[str, dict[str, Any]]
            ] = typing.cast(
                Optional[dict[str, dict[str, Any]]], params.get("prevalences")
            )
            if configured_prevalences is None:
                raise Exception("Can't load prevalences, no config was given.")
            for (
                bin_idx_string,
                bin_prevalence_params,
            ) in configured_prevalences.items():
                bin_idx = int(bin_idx_string)
                if bin_idx > num_total_bins:
                    raise Exception(
                        f"Configured bin id is higher than total of bins ({bin_idx}/{num_total_bins}"
                    )

                bin_prevalence_array = self.prepare_prevalence_array(
                    bin_idx, bin_prevalence_params, num_sample_groups
                )
                self.target_prevalences[bin_idx] = bin_prevalence_array

    @staticmethod
    def prepare_prevalence_array(
        bin_idx: int,
        bin_prevalence_params: dict[str, Any],
        num_sample_groups: int,
    ) -> list[int]:
        """Prepares the prevalences array, loading it from config, and filling repetitions if needed."""
        prevalence_array: list[int] = typing.cast(
            list[int], bin_prevalence_params.get("percentages_by_sample_group")
        )
        if prevalence_array is None:
            raise Exception(
                "Can't load prevalence percentages, no configuration was supplied"
            )
        prevalence_repeat: bool = typing.cast(
            bool, bin_prevalence_params.get("prevalence_repeat", False)
        )
        if prevalence_repeat:
            num_init_prevs = len(prevalence_array)
            random_rep_range: list[int] = typing.cast(
                list[int],
                bin_prevalence_params.get("prevalence_repeat_range", []),
            )
            print_and_log(
                f"Generating repetitions for sample groups (total sample groups: {num_sample_groups})"
            )
            full_prev_array = prevalence_array.copy()
            for i in range(num_init_prevs, num_sample_groups, num_init_prevs):
                repetition_prev_array = [
                    prevalence
                    + random.randrange(random_rep_range[0], random_rep_range[1])
                    for prevalence in prevalence_array
                ]
                full_prev_array += repetition_prev_array
            prevalence_array = full_prev_array
        print_and_log(
            f"Target prevalences for bin {bin_idx}: {prevalence_array}"
        )

        return prevalence_array


class SampleGroupSampleCounter:
    """Tracks how many samples have been added from each bin for the current sample group."""

    sample_group_id = -1
    samples_by_bin_counter: list[int] = []
    num_total_bins = 0
    sample_group_size = 0
    target_prevalences_by_bin: dict[str, Any] = {}

    def reset_if_needed(
        self,
        new_sample_group_id: int,
        num_total_bins: int,
        sample_group_size: int,
        target_prevalences,
    ):
        """Initialize the sample tracker if we changed sample groups."""
        if new_sample_group_id != self.sample_group_id:
            self.samples_by_bin_counter = [0] * num_total_bins
            self.sample_group_id = new_sample_group_id
            self.num_total_bins = num_total_bins
            self.sample_group_size = sample_group_size

            for bin_idx in target_prevalences:
                if self.sample_group_id > len(target_prevalences[bin_idx]):
                    raise Exception(
                        f"Configured bin prevalences not properly set up for sample group ({self.sample_group_id}"
                    )
                self.target_prevalences_by_bin[bin_idx] = target_prevalences[
                    bin_idx
                ][self.sample_group_id]

    def update(self, next_bin_id: int):
        """Updates the selected sample"""
        self.samples_by_bin_counter[next_bin_id] += 1

    def get_num_samples(self, bin_idx: int) -> int:
        """Returns the number of samples currently collected by bin idx given."""
        if bin_idx > len(self.samples_by_bin_counter):
            raise Exception(
                f"Invalid bin index retrieving number of samples: {bin_idx}"
            )
        return self.samples_by_bin_counter[bin_idx]

    def get_curr_prevalences(self) -> list[float]:
        """Returns an array with the % of prevalences for the samples currently selected, by bin."""
        curr_prevalences = [
            round(
                (100 * self.get_num_samples(bin_idx)) / self.sample_group_size,
                2,
            )
            for bin_idx in range(self.num_total_bins)
        ]
        return curr_prevalences

    def get_sample_group_target_prevalences(self) -> dict[str, Any]:
        """Gets the target prevalences by bin for the current sample group."""
        return self.target_prevalences_by_bin


# Global storage for configured target prevalences.
configured_prevalences = TargetPrevalences()

# Global counter for samples per bin for current sample group.
sample_counter = SampleGroupSampleCounter()


def get_bin_index(
    sample_index, sample_group_id, curr_bin_idx, num_total_bins, params
):
    """Implements the interface. Selects samples to have a given prevalence between options."""
    # Load and init the target prevalences arrays by bin, if needed (only done once).
    global configured_prevalences
    configured_prevalences.load_prevalence_arrays(params, num_total_bins)

    # Initialize the sample tracker each time we change sample groups (once per sample group).
    global sample_counter
    sample_group_size = params.get("sample_group_size")
    sample_counter.reset_if_needed(
        sample_group_id,
        num_total_bins,
        sample_group_size,
        configured_prevalences.get_all_prevalences(),
    )

    # Get the next bin id, ensuring we don't go over the target prevalences.
    print_and_log(f"Sample group id: {sample_group_id}")
    next_bin_id = get_random_bin_with_prevalence(
        num_total_bins, sample_counter.get_sample_group_target_prevalences()
    )

    # Update internal tracker and return.
    sample_counter.update(next_bin_id)
    return next_bin_id


def get_random_bin_with_prevalence(num_total_bins, target_prevalences_by_bin):
    """Gets next bin, getting first from the target prevalence ones, and then randomly."""
    curr_prevalences = sample_counter.get_curr_prevalences()
    print_and_log(
        f"Curr prevalences: {curr_prevalences}, Target prevalences: {target_prevalences_by_bin}"
    )

    # Select next bin that has a targeted prevalence not yet reached.
    next_bin_id = None
    for bin_idx in target_prevalences_by_bin.keys():
        if curr_prevalences[bin_idx] < target_prevalences_by_bin[bin_idx]:
            print_and_log(f"Selecting from targeted bin with idx {bin_idx}.")
            next_bin_id = bin_idx
            break

    if next_bin_id is None:
        print_and_log(
            f"Selecting random bin with exclusions {target_prevalences_by_bin.keys()}"
        )
        next_bin_id = randrange_with_exclusions(
            num_total_bins, target_prevalences_by_bin.keys()
        )

    return next_bin_id


def randrange_with_exclusions(range_max, exclusions):
    """Recursive function to get random values from a range avoiding excluded items."""
    selection = random.randrange(range_max)
    return (
        randrange_with_exclusions(range_max, exclusions)
        if selection in exclusions
        else selection
    )


def test(full_dataset: DataSet, drift_params: dict[str, Any]):
    """Tests that the given dataset was properly drifted."""
    bin_params: list[list[Any]] = typing.cast(
        list[list[Any]], drift_params.get("bins")
    )
    if bin_params is None:
        raise Exception("No bin configuration provided")
    num_total_bins = len(bin_params)
    configured_prevalences.load_prevalence_arrays(drift_params, num_total_bins)

    sample_group_size = typing.cast(
        int, drift_params.get("sample_group_size", 1)
    )
    num_samples = full_dataset.get_number_of_samples()
    num_sample_groups = int(num_samples / sample_group_size)

    # TODO: Change this to support bins values other than by dataset results?
    labelled_output = full_dataset.get_model_output()
    all_sample_group_prevalences: dict[int, dict[int, int]] = {}
    for curr_sample_group_id in range(0, num_sample_groups):
        # For a given sample group, count percentage of samples by result.
        curr_sample_group_starting_idx = (
            curr_sample_group_id * sample_group_size
        )
        sample_group_samples = labelled_output[
            curr_sample_group_starting_idx : curr_sample_group_starting_idx
            + sample_group_size
        ]
        unique, counts = numpy.unique(sample_group_samples, return_counts=True)
        prevalences = numpy.round((100 * counts) / sample_group_size, 2)
        sample_group_prevalences = dict(zip(unique, prevalences))

        print(
            f"Sample group {curr_sample_group_id} real prevalences: {sample_group_prevalences}"
        )
        all_sample_group_prevalences[
            curr_sample_group_id
        ] = sample_group_prevalences

    prevalences_by_bin: dict[int, list[int]] = {}
    for (
        _,
        sample_group_prevalences,
    ) in all_sample_group_prevalences.items():
        print(sample_group_prevalences, flush=True)
        for bin_idx, bin_prevalence in sample_group_prevalences.items():
            if bin_idx not in prevalences_by_bin:
                prevalences_by_bin[bin_idx] = []
            prevalences_by_bin[bin_idx].append(bin_prevalence)

    print(
        f"Expected prevalences by sample group and bin: {configured_prevalences.get_all_prevalences()}"
    )
    print(f"Obtained prevalences by sample group and bin: {prevalences_by_bin}")
