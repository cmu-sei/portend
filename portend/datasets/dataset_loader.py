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

import typing
from typing import Any, Optional

from portend.datasets.dataset import DataSet
from portend.datasets.ref_dataset import RefDataSet
from portend.utils.module import load_class

DEFAULT_DATASET_CLASS = "portend.datasets.dataset.DataSet"


def load_dataset_class(
    dataset_class_name: Optional[str],
) -> typing.Type[DataSet]:
    if dataset_class_name is None:
        dataset_class_name = DEFAULT_DATASET_CLASS
    return load_class(dataset_class_name)


def load_dataset(
    dataset_config: dict[str, str], dataset_filename: Optional[str] = None
) -> DataSet:
    """Load dataset from a file and class name."""

    # First load the class of dataset we will be using.
    dataset_class = load_dataset_class(dataset_config.get("dataset_class"))

    # If we are not using a base file, just load the file. If we are using one, we'll need to load the base and apply the reference dataset.
    if dataset_filename is None:
        dataset_filename = dataset_config.get("dataset_file")
    dataset_file_base = dataset_config.get("dataset_file_base")
    if dataset_file_base is None:
        dataset_instance: DataSet = dataset_class()
        dataset_instance.load_from_file(dataset_filename, dataset_config)
        return dataset_instance
    else:
        # First load the base dataset, which should be larger but contains the complete samples we need (and more).
        base_dataset: DataSet = dataset_class()
        base_dataset.load_from_file(dataset_file_base, dataset_config)

        # Now create a real, reconstructed dataset from the full samples in the base_dataset and the references in the dataset_filename reference dataset file.
        reconstructed_dataset = _load_full_from_ref_and_base(
            ref_dataset_file=dataset_filename,
            base_dataset=base_dataset,
            output_dataset_class=dataset_class,
            dataset_config=dataset_config,
        )
        return reconstructed_dataset


def _load_full_from_ref_and_base(
    ref_dataset_file: Optional[str],
    base_dataset: DataSet,
    output_dataset_class: type,
    dataset_config: dict[str, Any],
) -> DataSet:
    """Given a base dataset and a file with a references to samples there, combine them into a reconstructed one."""
    if ref_dataset_file is None:
        raise Exception("Can't load reference dataset, no filename provided")

    # First load all references into a RefDataSet.
    print("Loading ref dataset...", flush=True)
    reference_dataset = RefDataSet()
    reference_dataset.load_from_file(ref_dataset_file, dataset_config)

    # Now create a new, reconstructed dataset by getting the full data for each sample from their reference and the actual data in the base dataset.
    print("Creating full dataset from both...", flush=True)
    reconstructed_dataset = reference_dataset.create_from_reference(
        base_dataset, output_dataset_class, dataset_config
    )

    return reconstructed_dataset
