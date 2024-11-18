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
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from portend.utils import files as file_utils
from portend.utils.config import Config
from portend.utils.logging import print_and_log

DEFAULT_FIELDS_KEY = "fields"
GEN_CONFIG_PREFIX = "01_helper"


class ScenarioField:
    """A field used with different values to define scenarios."""

    def __init__(self, field_dict: dict[str, Any]):
        """Constructor from dict."""
        self.full_key = str(field_dict["key"])
        self.min = float(field_dict["min"])
        self.max = float(field_dict["max"])
        self.step = float(field_dict["step"])
        self.range = self._calculate_range()

    @staticmethod
    def load_scenario_fields(
        fields: list[dict[str, str]]
    ) -> list[ScenarioField]:
        """Loads a set of scenario fields from a dict into a list of ScenarioFields."""
        scenario_fields: list[ScenarioField] = []
        for field_dict in fields:
            scenario_fields.append(ScenarioField(field_dict))
        return scenario_fields

    def _calculate_range(self) -> npt.NDArray[np.float_]:
        """Generates a range based on the min, max and step for this field."""
        range = typing.cast(
            npt.NDArray[np.float_],
            np.arange(self.min, self.max + self.step, self.step),
        )
        return range

    def __str__(self) -> str:
        return f"Key: {self.full_key}, range: {self.range}"


def generate_config_files(
    base_file: str, output_folder: str, fields: list[dict[str, str]]
) -> None:
    """
    Generates a set of config files on the given folder, based on the given config by modifying the provided fields.

    :param base_file: The path to the template config file to use as a source and to modify into a new one.
    :param output_folder: The path to the folder to put the modified configuration files into.
    :param fields: A list of dictionaries with the fields to modify and their ranges. Each dictionary has four key/value pairs:
        - "key": the fully qualified path to get to the key while navigating the JSON structure of the base file.
        - "min", "max", "step": three float values to indicate the range of values to change the previous "key" to. For each valid value
                                in this range, a new configuration file will be created and stored.
        Example item in the list:
        {
            "key": "drift_scenario.params.vra",
            "min": "100", "max": "500", "step": "100"
        }
    """
    base_config = Config.load(base_file)

    # Recreate folder for output configs.
    file_utils.recreate_folder(output_folder)

    # Load scenario fields.
    scenario_fields = ScenarioField.load_scenario_fields(fields)

    for scenario_field in scenario_fields:
        # Iterate until we get to the actual field.
        parent_dict, field_sub_key = base_config.find_field(
            scenario_field.full_key
        )

        # Generate one file for each value in the range.
        value: float
        base_filename = Path(base_config.config_filename).stem
        range = scenario_field.range
        for i, value in enumerate(range):
            # Modify value for current case.
            parent_dict[field_sub_key] = value
            file_suffix = f"{field_sub_key}-{value}"

            # Save the new config file.
            file_num = f"{i}".zfill(3)  # Pad 3 zeroes if needed.
            new_file_name = f"{file_num}-{base_filename}--{file_suffix}.json"
            print_and_log(
                f"Generating file for {field_sub_key} value {value} named {new_file_name}"
            )
            base_config.config_filename = str(
                Path(output_folder) / new_file_name
            )
            base_config.save()


def load_scenario_field(
    fields_conf_file_path: str, fields_key: str = DEFAULT_FIELDS_KEY
) -> ScenarioField:
    """
    Loads information about the ScenarioField configued in the given generation config file, from the fields in it.

    :param fields_conf_file_path: The path to a config file with the fields to generate new config files.
    :param fields_key: The key to get from the file above where the fields are defined.

    :return: A ScenarioField with the information about the field and range to generate.
    """
    # Load field data.
    generated_config = Config.load(fields_conf_file_path)
    scenario_fields = ScenarioField.load_scenario_fields(
        generated_config.get(fields_key)
    )
    if len(scenario_fields) == 0:
        raise Exception("No fields for config file generation were found.")

    # Assume for now we support only one field.
    if len(scenario_fields) > 1:
        raise Exception(
            "Multiple fields are not supported for config file generation."
        )
    scenario_field = scenario_fields[0]
    print_and_log(f"Scenario field found: <{scenario_field.full_key}>")
    return scenario_field
