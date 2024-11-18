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

import time


class Timer:
    """Simple timer class that times wall time and process time in milliseconds."""

    S_TO_MS_FACTOR = 1000

    def __init__(self):
        self.start_time: float
        self.proc_start_time: float
        self.elapsed: float
        self.proc_elapsed: float

    def start(self):
        """Starts tracking time."""
        self.start_time = time.time()
        self.proc_start_time = time.process_time()

    def stop(self):
        """Stops tracking time."""
        end_time = time.time()
        proc_end_time = time.process_time()

        self.elapsed = (end_time - self.start_time) * self.S_TO_MS_FACTOR
        self.proc_elapsed = (
            proc_end_time - self.proc_start_time
        ) * self.S_TO_MS_FACTOR
