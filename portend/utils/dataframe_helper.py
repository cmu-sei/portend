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

from pathlib import Path

import pandas as pd

from portend.utils import files as file_utils

# Dataframe handler helper functions.


def merge_files(file1: str, file2: str, output_filename: str):
    """Merges data from two JSON files into one."""
    dataframe1 = load_dataframe_from_file(file1)
    dataframe2 = load_dataframe_from_file(file2)

    print("Merging DataFrames", flush=True)
    merged_df = pd.concat([dataframe1, dataframe2])

    save_dataframe_to_file(merged_df, output_filename)


def load_dataframe_from_file(filename: str) -> pd.DataFrame:
    """Loads a JSON file into a dataframe, with default params and log output."""
    print("Loading input file: " + filename, flush=True)
    if not Path(filename).exists():
        raise IOError(f"Dataframe on path {filename} does not exist.")
    data_df = pd.read_json(filename)
    print("Done loading data. Rows: " + str(data_df.shape[0]), flush=True)
    return data_df


def save_dataframe_to_file(dataframe: pd.DataFrame, filename: str):
    """Stores a pandas dataframe to a JSON file, with default params and log output."""
    # Ensure output folder exists.
    file_utils.create_folder_for_file(filename)

    print(
        "Saving DataFrame to JSON file "
        + filename
        + " (rows: "
        + str(dataframe.shape[0])
        + ")",
        flush=True,
    )
    json_data = dataframe.to_json(
        orient="records", indent=4, date_format="epoch", date_unit="ms"
    ).replace("\\/", "/")
    with open(filename, "w") as output_file:
        output_file.write(json_data)
    print(f"Finished saving JSON file: {filename}", flush=True)
