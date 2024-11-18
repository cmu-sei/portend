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

from portend.analysis.time_series.timeseries import TimeSeries
from portend.metrics.ts_metrics import ErrorMetric


class ZTestMetric(ErrorMetric):
    # Overriden.
    def metric_error(
        self,
        time_interval_id: int,
        time_series: TimeSeries,
        ts_predictions: TimeSeries,
    ):
        """Calculates the STEPD error."""
        x1 = time_series.get_aggregated(time_interval_id)
        x2 = ts_predictions.get_pdf_params(time_interval_id).get("mean")
        sigma = ts_predictions.get_pdf_params(time_interval_id).get("std_dev")
        t = (x1 - x2) / sigma
        return t
