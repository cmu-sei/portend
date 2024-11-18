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

from typing import Any

import pandas as pd
from statsmodels.regression.linear_model import PredictionResults
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults

from portend.analysis.time_series.timeseries import TimeSeries
from portend.models.ml_model import MLModel


class TimeSeriesModel(MLModel):
    """Base class for ARIMA-based Time Series models to be used with the system."""

    model: ARIMAResults

    # Implemented.
    def create_model(
        self, time_intervals, aggregated_history, interval_unit, params
    ):
        """Creates the model internally."""
        # Prepare data as dataframe.
        data = {"Date": time_intervals, "Values": aggregated_history}
        dataframe = pd.DataFrame(data)
        dataframe.set_index("Date", inplace=True)

        # Rebuild index to avoid warning and issue with frequency.
        tidx = pd.DatetimeIndex(dataframe.index.values, freq=interval_unit)
        dataframe.set_index(tidx, inplace=True)
        print(dataframe)

        # Build and fit model.
        model_order = (
            int(params.get("order_p")),
            int(params.get("order_q")),
            int(params.get("order_d")),
        )
        model = ARIMA(dataframe, freq=interval_unit, order=model_order)
        fit_model: ARIMAResults = model.fit()
        self.model = fit_model

    # Implemented.
    def save_to_file(self, model_filename: str):
        self.model.save(model_filename)

    # Implemented.
    def load_from_file(self, model_filename: str):
        self.model = ARIMAResults.load(model_filename)

    # Implemented.
    def predict(
        self, input: TimeSeries
    ) -> tuple[TimeSeries, dict[str, dict[str, Any]]]:
        """Creates the prediction data, for now only pdf params, based on the fit model."""
        intervals = input.get_time_intervals()
        start_interval = intervals[0]
        end_interval = intervals[len(intervals) - 1]

        # Predict the pdf params for each time interval in the series.
        pdf_params = _get_prediction_params(
            self.model, start_interval, end_interval
        )

        # Return a time series object with the same intervals plus the pdf params set for each.
        ts_predictions = TimeSeries()
        ts_predictions.set_time_intervals(input.get_time_intervals())
        ts_predictions.set_pdf_params(pdf_params)
        return ts_predictions, {}


def _get_prediction_params(
    fit_model: ARIMAResults,
    start_interval: pd.Timestamp,
    end_interval: pd.Timestamp,
) -> list[Any]:
    """Gets the prediction params for the given intervals."""
    print(f"Intervals to predict: {start_interval} to {end_interval}")
    forecasts: PredictionResults = fit_model.get_prediction(
        start=start_interval, end=end_interval
    )

    pdf_params = []
    confidence_level = 0.68
    alpha = 1 - confidence_level
    summary = forecasts.summary_frame(alpha=alpha)
    print(summary)
    for _, row in summary.iterrows():
        mu = row["mean"]
        sigma = row["mean_ci_upper"] - mu
        pdf_params.append({"mean": mu, "std_dev": sigma})

    return pdf_params
