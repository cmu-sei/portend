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

from portend.models.process_model import ProcessModel
from portend.utils import process

DEFAULT_CONTAINER_NAME = "portend-process-container"
CONTAINER_BASH = "portend/models/run_container_process.sh"
DEFAULT_GPUS = "none"
DEFAULT_PATH = "./"


class ProcessContainerModel(ProcessModel):
    """Class that represents an external process model running in a container."""

    # Implemented.
    def load_additional_params(self, data: dict[str, str]):
        super(ProcessContainerModel, self).load_additional_params(data)

        self.container_name = data.get(
            "model_container_name", DEFAULT_CONTAINER_NAME
        )
        self.container_params = data.get("model_container_params", "")
        self.container_io_path = data.get(
            "model_container_io_path", DEFAULT_PATH
        )
        self.container_gpus = data.get("model_container_gpus", DEFAULT_GPUS)

        # Get the full path for the process io folder, so we can pass it to the container to mount.
        host_abs_path = os.getcwd()
        if process.is_running_in_docker():
            host_abs_path = os.environ.get("BASE_PATH", "")
        host_io_path = os.path.join(host_abs_path, self.RELATIVE_IO_FOLDER)

        # Build command.
        self.command = [
            "bash",
            "-x",
            CONTAINER_BASH,
            host_io_path,
            self.container_io_path,
            self.container_gpus,
            self.container_name,
        ]
        self.command.extend(self.container_params.split())
        print(f"Container command to run external algorithm: {self.command}")
