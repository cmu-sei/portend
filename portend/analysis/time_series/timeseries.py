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

import math
import typing
from typing import Any, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd

from portend.utils.logging import print_and_log


class IntervalGenerator:
    """Abstract class with common functions for interval generators."""

    def __init__(self, starting_interval: pd.Timestamp, interval_unit: str):
        self.starting_interval = starting_interval
        self.interval_unit = interval_unit

    def calculate_time_interval(self, sample_idx: int) -> pd.Timestamp:
        """Calculates the time interval number where a sample falls into, given the time step."""
        raise NotImplementedError()

    def get_last_time_interval(self) -> pd.Timestamp:
        raise NotImplementedError()

    def get_number_of_intervals(self) -> int:
        """Returns the number of intervals for the total of samples."""
        interval_delta: int = (
            self.get_last_time_interval()
            - self.starting_interval
            + pd.to_timedelta(1, self.interval_unit)  # type: ignore
        )
        return int(interval_delta / pd.to_timedelta(1, self.interval_unit))  # type: ignore


class TimestampIntervalGenerator(IntervalGenerator):
    """Calculates time intervals based on timestamps."""

    def __init__(
        self,
        starting_interval: pd.Timestamp,
        interval_unit: str,
        timestamps: npt.NDArray[np.int_],
    ):
        super().__init__(starting_interval, interval_unit)
        self.timestamps = timestamps

    def calculate_time_interval(self, sample_idx: int) -> pd.Timestamp:
        """Calculates the time interval number where a timestamp falls into, given the time step."""
        min_timestamp = self.timestamps[0]
        delta = math.floor(
            pd.to_timedelta((self.timestamps[sample_idx] - min_timestamp))
            / pd.to_timedelta(1, self.interval_unit)  # type: ignore
        )
        interval: pd.Timestamp = self.starting_interval + pd.Timedelta(
            delta, unit=self.interval_unit  # type: ignore
        )
        return interval

    def get_last_time_interval(self) -> pd.Timestamp:
        return self.calculate_time_interval(self.timestamps.size - 1)


class NumSamplesIntervalGenerator(IntervalGenerator):
    """Calculates time intervals based on number of samples per interval."""

    def __init__(
        self,
        starting_interval: pd.Timestamp,
        interval_unit: str,
        samples_per_time: int,
        num_samples: int,
    ):
        super().__init__(starting_interval, interval_unit)
        self.samples_per_time = samples_per_time
        self.num_samples = num_samples

    def calculate_time_interval(self, sample_idx: int) -> pd.Timestamp:
        """Calculates the time interval number where a sample falls into, given the time step."""
        delta = math.floor(sample_idx / self.samples_per_time)
        interval: pd.Timestamp = self.starting_interval + pd.Timedelta(
            delta, unit=self.interval_unit  # type: ignore
        )
        return interval

    def get_last_time_interval(self) -> pd.Timestamp:
        return self.calculate_time_interval(self.num_samples - 1)


class TimeSeries:
    """Represents a time series with a time interval and an aggregated value."""

    time_intervals: list[pd.Timestamp] = []
    aggregated = np.empty(0)
    num_samples: npt.NDArray[np.int_] = np.empty(0, dtype=int)
    pdf: list[npt.NDArray[Any]] = []
    pdf_params: list[Any] = []

    def check_valid_idx(self, time_interval_idx: int):
        """Checks if the interval idx is inside the valid range"""
        if (
            time_interval_idx >= self.get_num_intervals()
            or time_interval_idx < 0
        ):
            raise Exception(
                f"Invalid time interval id passed: {time_interval_idx}, length is {self.get_num_intervals()}"
            )

    def set_time_intervals(self, time_intervals: list[Any]):
        """Sets the time intervals list."""
        self.time_intervals = time_intervals

    def get_time_intervals(self) -> list[pd.Timestamp]:
        """Getter for the list of time intervals."""
        return self.time_intervals

    def get_num_intervals(self) -> int:
        """Returns the amount of intervals in the object."""
        return len(self.time_intervals)

    def get_interval_index(self, interval) -> int:
        """Returns the index of the given interval."""
        interval_idx = self.time_intervals.index(interval)
        return interval_idx

    def get_aggregated(
        self, time_interval_idx: Optional[int] = None
    ) -> npt.NDArray[Any]:
        """Returns either the full array of aggregated values, or a specific one by the time interval index."""
        if time_interval_idx is None:
            return self.aggregated
        else:
            self.check_valid_idx(time_interval_idx)
            return typing.cast(
                npt.NDArray[Any], self.aggregated[time_interval_idx]
            )

    def get_pdf(self, time_interval_idx: int) -> float:
        """Getter for a specific pdf for the given time interval."""
        self.check_valid_idx(time_interval_idx)
        return float(self.pdf[time_interval_idx])

    def set_pdf(self, pdf: list[npt.NDArray[Any]]):
        """Setter for the list of pdfs."""
        self.pdf = pdf

    def get_pdf_params(self, time_interval_idx: int) -> Any:
        """Getter for a specific pdf_params for the given time interval."""
        self.check_valid_idx(time_interval_idx)
        return self.pdf_params[time_interval_idx]

    def set_pdf_params(self, pdf_params: list[Any]):
        """Setter for the list of pdf parameters."""
        self.pdf_params = pdf_params

    def get_num_samples(self, time_interval_idx: int) -> int:
        """Returns the number of samples aggregated for the given time interval."""
        self.check_valid_idx(time_interval_idx)
        return int(self.num_samples[time_interval_idx])

    def get_model_inputs(self):
        """Returns the time intervals and aggregated values as a model input."""
        return [self.get_time_intervals(), self.get_aggregated()]

    def setup_time_intervals(
        self, starting_interval: pd.Timestamp, unit: str, num_intervals: int
    ):
        """Sets up time intervals given the starting one, unit, and how many, as well as pre-allocating other arrays."""
        self.time_intervals = [pd.Timestamp()] * num_intervals
        for i in range(0, num_intervals):
            self.time_intervals[i] = starting_interval + pd.Timedelta(
                i, unit=unit  # type: ignore
            )

        self.aggregated = np.zeros(self.get_num_intervals(), dtype=int)
        self.num_samples = np.zeros(self.get_num_intervals(), dtype=int)
        self.pdf = [np.empty(1)] * self.get_num_intervals()
        self.pdf_params = [{}] * self.get_num_intervals()

    def aggregate(
        self,
        starting_interval: pd.Timestamp,
        interval_unit: str,
        values: npt.NDArray[np.float_],
        interval_generator: IntervalGenerator,
    ):
        """Aggregates a given dataset, and stores it in memory."""
        # Pre-allocate space, and fill up times, given timestamps in dataset.
        num_intervals = interval_generator.get_number_of_intervals()
        self.setup_time_intervals(
            starting_interval, interval_unit, num_intervals
        )
        print_and_log(
            f"Last time interval: {self.time_intervals[len(self.time_intervals) - 1]}"
        )

        # Go over all samples, adding their output to the corresponding position in the aggregated array.
        total_num_samples = values.size
        for sample_idx in range(0, total_num_samples):
            # Calculate the interval for the current sample.
            sample_time_interval = interval_generator.calculate_time_interval(
                sample_idx
            )
            # print(f"Sample: {sample_idx}, time interval: {sample_time_interval}")

            # Update the aggregated sum and the number of samples.
            interval_idx = self.get_interval_index(sample_time_interval)
            self.aggregated[interval_idx] += values[sample_idx]
            self.num_samples[interval_idx] += 1

        print_and_log("Finished aggregating.")

    def aggregate_by_timestamp(
        self,
        start_interval_string: Optional[str],
        interval_unit: Optional[str],
        values: npt.NDArray[Any],
        timestamps: npt.NDArray[np.int_],
    ):
        """Aggregates a given dataset on the given time step, and stores it in memory."""
        if start_interval_string is None or interval_unit is None:
            raise Exception("Interval or unit not configured.")
        print_and_log("Aggregating by timestamp")
        start_interval = pd.to_datetime(start_interval_string)
        interval_generator = TimestampIntervalGenerator(
            start_interval, interval_unit, timestamps
        )
        return self.aggregate(
            start_interval, interval_unit, values, interval_generator
        )

    def aggregate_by_number_of_samples(
        self,
        start_interval_string: Optional[str],
        interval_unit: Optional[str],
        values: npt.NDArray[Any],
        samples_per_time: int,
    ):
        """Aggregates a given dataset on the given time step, and stores it in memory."""
        if start_interval_string is None or interval_unit is None:
            raise Exception("Interval or unit not configured.")
        print_and_log("Aggregating by number of samples")
        start_interval = pd.to_datetime(start_interval_string)
        interval_generator = NumSamplesIntervalGenerator(
            start_interval, interval_unit, samples_per_time, values.size
        )
        return self.aggregate(
            start_interval, interval_unit, values, interval_generator
        )

    def to_dict(self) -> dict[str, Any]:
        """Returns the main attributes of this object as a dictionary."""
        dictionary = {
            "time_intervals": self.time_intervals,
            "aggregated": self.aggregated,
            "num_samples": self.num_samples,
            "pdf": self.pdf,
            "pdf_params": self.pdf_params,
        }
        return dictionary


def create_test_time_series(
    dist_start: int = 0, dist_end: int = 10, dist_total: int = 10
):
    """A manually created time series for testing."""

    samples_per_time = 4
    output_values = np.array([1, 2, 1, 3, 4, 5, 6, 1, 2, 1, 3, 4, 5, 6, 1])

    time_series = TimeSeries()
    time_series.aggregate_by_number_of_samples(
        "2021-11-01",
        "days",
        output_values,
        samples_per_time,
    )
    time_series.set_pdf(
        [np.random.randint(dist_start, dist_end, (dist_total))]
        * time_series.get_num_intervals()
    )
    time_series.set_pdf_params(
        [{"mean": 5, "std_dev": 3}] * time_series.get_num_intervals()
    )

    return time_series


def test():
    """Quick test of a timeseries aggregation."""
    print("Testing")

    expected_values = np.array([7, 16, 10, 12])
    time_series = create_test_time_series()
    print(time_series.to_dict())
    print(f"Expected values: {expected_values}")


if __name__ == "__main__":
    test()
