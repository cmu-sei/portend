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

import importlib
import types
import typing
from typing import Optional

from portend.utils.logging import print_and_log


def load_module(
    module_path: typing.Optional[str], base_package: str = ""
) -> types.ModuleType:
    """Loads the given module dynamically."""
    if module_path is None:
        raise Exception("Did not receive module path to load.")

    try:
        full_module_path = base_package + module_path
        print_and_log("Importing module " + full_module_path)
        module = importlib.import_module(full_module_path)
    except ModuleNotFoundError as e:
        print_and_log(
            "Could not find or load module: "
            + full_module_path
            + ". Aborting. Error: "
            + str(e)
        )
        exit(1)

    return module


def load_class(class_path: Optional[str]) -> typing.Type[typing.Any]:
    """
    Returns a class type of the given class name/path.
    :param class_path: A path to a class to use, including absolute package/module path and class name.
    """
    if class_path is None:
        raise Exception("No class name was received to load.")

    # Split into package/module and class name.
    print_and_log(f"Loading class: {class_path}")
    parts = class_path.rsplit(".", 1)
    module_name = parts[0]
    class_name = parts[1]

    loaded_module = importlib.import_module(module_name)
    class_type = typing.cast(
        typing.Type[typing.Any], getattr(loaded_module, class_name)
    )
    return class_type
