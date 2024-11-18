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

import datetime
import json
import os
import shutil
import typing
from pathlib import Path
from typing import Any, Dict

from portend.utils.logging import print_and_log


def _get_timestamped_name(file_path: str, prefix: str) -> str:
    """Returns a time-stamped name based on the give filename and prefix, in the same path."""
    folder = os.path.dirname(file_path)
    parts = os.path.splitext(os.path.basename(file_path))
    descriptor = parts[0]
    extension = parts[1]
    drift_file_name = (
        prefix
        + "-"
        + descriptor
        + "-"
        + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    )
    return os.path.join(folder, drift_file_name + "." + extension)


def save_timestamped_backup(file_path: str, prefix: str):
    """Saves a copy of the file pointed at by file_path, to a file with a timestamped name in the same location."""
    print_and_log("Copying output file to timestamped backup.")
    time_stamped_file_path = _get_timestamped_name(file_path, prefix)
    shutil.copyfile(file_path, time_stamped_file_path)


def create_folder_for_file(filepath: str):
    """Creates the folder tree for a given file path, if needed."""
    folder = Path(filepath).parent
    if not folder.exists():
        folder.mkdir(parents=True)


def recreate_folder(folder_path: str):
    """Removes (if needed) and creates given folder."""
    if Path(folder_path).exists():
        shutil.rmtree(folder_path)
    Path(folder_path).mkdir(parents=True)


def save_dict_to_json_file(
    data: dict[Any, Any], output_path: str, data_name: str = "data"
):
    print_and_log(f"Saving {data_name} to JSON file : {output_path}")
    create_folder_for_file(output_path)
    with open(output_path, "w") as outfile:
        json.dump(data, outfile, indent=4)
    print_and_log("Finished saving JSON file")


def load_json_file_to_dict(
    file_path: str, data_name: str = "data"
) -> dict[str, Any]:
    """Loads data from a JSON file into a dict."""
    print_and_log(f"Loading {data_name} from JSON file {file_path}")
    with open(file_path, "r") as infile:
        data = typing.cast(Dict[str, Any], json.load(infile))
    return data
