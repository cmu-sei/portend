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

import argparse
import json
from typing import Any, Optional

from portend.utils.logging import print_and_log


class Config:
    """Handles a JSON configuration."""

    config_data: dict[str, Any] = {}
    config_filename = ""

    @staticmethod
    def load(config_filename: str) -> Config:
        """Loads config data from the given file."""
        config = Config()
        with open(config_filename) as config_file:
            config.config_data = json.load(config_file)
        config.config_filename = config_filename
        return config

    def get(self, key_name: str, default: Optional[Any] = None):
        """Returns a dict with the default values."""
        if key_name in self.config_data:
            return self.config_data[key_name]
        else:
            return default

    def contains(self, key_name: str):
        """Checks whether a specific config is there."""
        return key_name in self.config_data

    def save(self):
        """Stores a config to disk."""
        with open(self.config_filename, mode="w") as config_file:
            json.dump(self.config_data, config_file, indent=4)

    @staticmethod
    def get_config_from_args(arguments: argparse.Namespace) -> Config:
        """Returns a Config object based on the params received."""
        config_file = None
        if arguments.c:
            print(f"Using config passed as argument: {arguments.c}")
            config_file = arguments.c
        else:
            raise RuntimeError("No valid config was supplied")

        return Config.load(config_file)

    def find_field(self, key: str) -> tuple[dict[str, Any], str]:
        """
        Finds the containing dict and subkey to access the given full-path field in this config.

        :param key: The key to find, as full path of keys separated by dots (e.g. "scenarios.params.alpha")
        :return: The value of the containing dict with the key, and the sub_key to get the value of the field inside that dict (e.g. the "params" dict, and "alpha" as sub-key)
        """
        # Split key into the nested subkeys.
        split_key = key.split(".")

        # Iterate until we get to the actual field.
        parent = self.config_data
        curr_key = key
        subconfig = self.config_data
        for sub_key in split_key:
            if sub_key not in subconfig:
                raise RuntimeError(
                    f"Subkey {sub_key} of key {key} was not found in config {self.config_filename}"
                )
            else:
                parent = subconfig
                curr_key = sub_key
                subconfig = subconfig[sub_key]

        print_and_log(f"Found sub key: {curr_key} inside parent dict: {parent}")
        return parent, curr_key
