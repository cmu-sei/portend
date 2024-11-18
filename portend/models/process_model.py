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

import os
import shutil
import typing
from typing import Any, List

import pandas as pd

from portend.models.ml_model import MLModel
from portend.utils import csv, process
from portend.utils.typing import SequenceLike

DEFAULT_INPUT_FOLDER = "input"
DEFAULT_OUTPUT_FILE = "output.csv"
DEFAULT_WORKING_DIR = "./"


class ProcessModel(MLModel):
    """Class that represents a model that is really an external process containing ML models inside it."""

    RELATIVE_IO_FOLDER = "./process_io"

    command: list[str] = []

    # Implemented.
    def load_additional_params(self, data: dict[str, str]):
        # In this case, the model_filename is actually the command to execute for the algorithm to run.
        model_command = (
            data.get("model_command") if "model_command" in data else None
        )
        self.input_folder = data.get("model_input_folder", DEFAULT_INPUT_FOLDER)
        self.output_file = data.get("model_output_file", DEFAULT_OUTPUT_FILE)
        self.working_dir = data.get("model_working_dir", DEFAULT_WORKING_DIR)
        self.extra_output_files: list[str] = typing.cast(
            List[str], data.get("model_extra_output_files", [])
        )

        if model_command is not None:
            print(f"Storing command to run external algorithm: {model_command}")
            self.command = model_command.split()

        # Add base IO folder to process IO folder and file paths.
        self.input_folder = os.path.join(
            self.RELATIVE_IO_FOLDER, self.input_folder
        )
        self.output_file = os.path.join(
            self.RELATIVE_IO_FOLDER, self.output_file
        )

        # Add host IO path to extra files.
        # NOTE: since self.extra_output_files just points to the array of file names in the config,
        # this actually ends up adding the full path to the array in the config, which can be convenient
        # but also confusing when used later.
        if len(self.extra_output_files) > 0:
            for position, extra_file in enumerate(self.extra_output_files):
                self.extra_output_files[position] = os.path.join(
                    self.RELATIVE_IO_FOLDER, extra_file
                )

    # Implemented.
    def predict(
        self, input: list[list[str]]
    ) -> tuple[SequenceLike, dict[str, dict[str, Any]]]:
        """
        Prediction works by setting the input for the external process, executing it, and processing the output file.
        Assumptions:
          - Input will be one or more lists of files to be passed to the process to the DEFAULT_INPUT_FOLDER folder.
          - Output will be in a CSV output file defined in the constant DEFAULT_OUTPUT_FILE.
          - Results will be returned as an list, and extra data as a dict keyed by filename, with a dict each keyed by column.
        """
        # TODO: This only support passing files. There is currently no support for other types of inputs.
        if len(self.command) == 0:
            raise Exception(
                "Command for external process model has not been set, can't predict."
            )
        if len(input) == 0:
            raise Exception("Input contains no data.")

        # We need to set the input for the external process.
        print(f"Copying files to input folder {self.input_folder}.")
        if os.path.exists(self.input_folder):
            shutil.rmtree(self.input_folder)
        os.makedirs(self.input_folder, exist_ok=True)
        for file_list in input:
            for filename in file_list:
                filename_only = os.path.basename(filename)
                updated_path = os.path.join(self.input_folder, filename_only)
                shutil.copy(filename, updated_path)
                print(f"Copied {filename} to {updated_path}")

        # Process will run and output will be csv files containing just the results.
        print(f"Executing external process: {self.command}")
        process.run_with_output(self.command, self.working_dir)
        print("Finished executing command.")

        # Load data and return it.
        print(f"Loading output from file: {self.output_file}")
        output = pd.read_csv(self.output_file, header=None).to_numpy()

        # If available, load additional output files' data.
        extra_output_data = csv.load_csv_data(self.extra_output_files)

        return output, extra_output_data
