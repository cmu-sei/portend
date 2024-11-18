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
import types
from typing import Any, Optional

import numpy as np
import pandas

from portend.datasets import ref_dataset
from portend.datasets.dataset import DataSet
from portend.drifts.temporal import databin
from portend.utils import module as module_utils
from portend.utils.logging import print_and_log
from portend.utils.typing import SequenceLike

TEMPORAL_DRIFT_PACKAGE = "portend.drifts.temporal.submodules."

# The configs applicable to all submodules of temporal drift are:
#
# - **submodule**: name of the specific submodule controlling the temporal drift, from the temporal folder.
# - **bin_shuffle**: (OPTIONAL) true or false to indicate whether to shuffle samples in each bin after sorting them. Defaults to true.
# - **bin_values**: (OPTIONAL) dataset values to use when sorting into bins. Current values: "all" (one bin with everything), "results" for results/output/truth values. Defaults to "results".
# - **bins**: array defining bins to split the **dataset** into when generating drift. Each item in the array will also be an array, containing first the bin name, and then the bin order (i.e, ["no_iceberg", 0])
# - **timestamps**: timestamp generation parameters for the drifted dataset.
#   - **start_datetime**: timestamp for the first element of the drifted dataset.
#   - **increment_unit**: increment unit for each new sample in the drifted dataset. Possible values available here: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
# - **max_num_samples**: how many samples to output into the drifted dataset.
# - **sample_group_size**: size of the sample group for generating the drifted dataset.
# - **sample_group_shuffle**: (OPTIONAL) true or false to indicate whether to shuffle samples in each group after selecting them from bins. Defaults to true.
#


def load_sub_module(submodule_name: Optional[str]) -> types.ModuleType:
    """Loads the drift submodule."""
    if submodule_name:
        raise Exception("Can't load submodule since it was not provided")
    return module_utils.load_module(submodule_name, TEMPORAL_DRIFT_PACKAGE)


def apply_drift(base_dataset: DataSet, params: dict[str, Any]) -> DataSet:
    """Applies drift on a given dataset"""
    print("Applying drift to generated drifted dataset.")
    max_num_samples = params.get("max_num_samples", 1)
    sample_group_size = params.get("sample_group_size", 1)
    shuffle_on = (
        params.get("sample_group_shuffle")
        if "sample_group_shuffle" in params
        else True
    )
    drift_submodule = load_sub_module(params.get("submodule"))

    # Setup bins.
    bin_value = params.get("bin_value", "results")
    bin_shuffle = params.get("bin_shuffle", True)
    input_bins = _load_bins(
        base_dataset, params.get("bins", []), bin_value, bin_shuffle
    )

    # Loop until we get all samples we want.
    drifted_dataset = ref_dataset.RefDataSet()
    curr_bin_offset = 0
    sample_group_id = 0
    new_samples: list[list[Any]] = []
    original_ids = []
    sample_group_ids = []
    while len(new_samples) < max_num_samples:
        print_and_log(
            f"Now getting data for sample group of size {sample_group_size}, using bin offset {curr_bin_offset}"
        )
        sample_group_sample_ids = _generate_sample_group_samples(
            drift_submodule,
            sample_group_id,
            curr_bin_offset,
            input_bins,
            sample_group_size,
            params,
        )

        # Randomize results in sample group to avoid stacking bin results at the end.
        if shuffle_on:
            print_and_log("Shuffling sample group samples.")
            random.shuffle(sample_group_sample_ids)

        # Gather new data in lists.
        for _ in range(len(sample_group_sample_ids)):
            new_samples.append(drifted_dataset.new_empty_sample())
        original_ids.extend(sample_group_sample_ids)
        sample_group_ids.extend(
            [sample_group_id] * len(sample_group_sample_ids)
        )

        # Offset to indicate the starting bin for the condition.
        curr_bin_offset = (curr_bin_offset + 1) % len(input_bins)
        sample_group_id += 1

    # Set the dataset with the data.
    drifted_dataset.create_from_lists(
        new_samples, original_ids, sample_group_ids
    )
    print_and_log("Finished applying drift")

    # Adds timestamps.
    _add_timestamps(drifted_dataset, params.get("timestamps", {}))

    return drifted_dataset


def _load_bins(
    base_dataset: DataSet,
    bin_params: list[list[Any]],
    bin_value: str = "results",
    shuffle: bool = True,
) -> list[databin.DataBin]:
    """Loads a dataset into bins"""
    if len(bin_params) == 0:
        raise Exception("No valid bins were configured.")

    # Sort into bins.
    print_and_log(f"Bins: {bin_params}")
    values = _get_bin_values(base_dataset, bin_value)
    bins = databin.create_bins(bin_params, shuffle)
    bins = databin.sort_into_bins(base_dataset.get_ids(), values, bins)

    # Setup queues.
    print_and_log("Filled bins: ")
    for curr_bin in bins:
        print_and_log(f" - {curr_bin.info()}")
        curr_bin.setup_queue()

    return bins


def _get_bin_values(base_dataset: DataSet, bin_value: str) -> SequenceLike:
    """Gets the values to be used when sorting into bins for the given dataset, from the configured options."""
    values: SequenceLike
    if bin_value == "results":
        values = base_dataset.get_model_output()
    elif bin_value == "all":
        # We set all values to 0, assuming single bin will also set its value to 0.
        values = [0] * base_dataset.get_number_of_samples()
    else:
        raise Exception(f"Invalid bin value configured: {bin_value}")
    return values


def _generate_sample_group_samples(
    drift_submodule: types.ModuleType,
    sample_group_id: int,
    curr_bin_offset: int,
    input_bins: list[databin.DataBin],
    sample_group_size: int,
    params: dict[str, Any],
) -> list[Any]:
    """Chooses samples for a given sample group size, and from the given current bin index."""

    # Get all values for this sample group.
    sample_group_sample_ids = []
    for sample_index in range(0, sample_group_size):
        bin_idx: int = drift_submodule.get_bin_index(
            sample_index,
            sample_group_id,
            curr_bin_offset,
            len(input_bins),
            params,
        )
        # print_and_log(f"Selecting from bin {bin_idx}")
        curr_bin = input_bins[bin_idx]

        if curr_bin.get_queue_length() == 0:
            print_and_log("No more items in queue, resetting it.")
            curr_bin.setup_queue()
        next_sample_id = curr_bin.pop_from_queue()
        sample_group_sample_ids.append(next_sample_id)
    return sample_group_sample_ids


def _add_timestamps(drifted_dataset: DataSet, timestamp_params: dict[str, str]):
    """Adds sequential timestamps to a dataset."""
    if len(timestamp_params):
        raise Exception("No timestamp config was received.")

    num_samples = drifted_dataset.get_number_of_samples()
    increment_unit = timestamp_params.get("increment_unit")
    start_datetime_config = timestamp_params.get("start_datetime")
    if not start_datetime_config or start_datetime_config == "auto":
        # TODO: Get first forecast timestamp from time series as first drift date.
        raise NotImplementedError(
            "Automatic start datetime functionality not implemented yet."
        )
    else:
        start_datetime = pandas.to_datetime(start_datetime_config)

    # Generate sequential timestamps for as many samples as we have, with the given start time and increment.
    print_and_log(
        f"Generating timestamps from starting time {start_datetime} and increment unit {increment_unit}"
    )
    timestamps: list[int] = []
    for i in range(0, num_samples):
        increment = pandas.to_timedelta(i, increment_unit)  # type: ignore
        timestamps.append(
            int(pandas.Timestamp(start_datetime + increment).timestamp())
        )
    drifted_dataset.set_timestamps(np.array(timestamps))
    print_and_log("Generated and applied timestamps.")
