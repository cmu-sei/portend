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

import csv
from typing import Any


def write_drone_csv_coordinates(
    coordinates: list[dict[str, Any]], output_path: str
):
    """Writes a CSV file with coordinates for each image."""
    header = [
        "Filename",
        "Latitude",
        "Longitude",
        "Altitude",
        "Gimball_Roll",
        "Gimball_Yaw",
        "Gimball_Pitch",
        "Flight_Roll",
        "Flight_Yaw",
        "Flight_Pitch",
    ]

    # Default values
    altitude = "NaN"
    gimball_roll = "NaN"
    gimball_yaw = "NaN"
    gimball_pitch = "NaN"
    flight_roll = "NaN"
    flight_yaw = "NaN"
    flight_pitch = "NaN"

    lines = []
    for info in coordinates:
        filename = info["filename"]
        lat = info["center_lat"]
        lon = info["center_long"]
        line = [
            filename,
            lat,
            lon,
            altitude,
            gimball_roll,
            gimball_yaw,
            gimball_pitch,
            flight_roll,
            flight_yaw,
            flight_pitch,
        ]
        lines.append(line)

    write_csv(output_path, lines, header)


def write_map_csv_coordinates(
    coordinates: list[dict[str, Any]], output_path: str
):
    """Writes a CSV file with coordinates for a map image."""
    header = [
        "Filename",
        "Top_left_lat",
        "Top_left_lon",
        "Bottom_right_lat",
        "Bottom_right_long",
    ]
    lines = []
    for info in coordinates:
        filename = info["filename"]
        top_left_lat = info["top_left_lat"]
        top_left_long = info["top_left_long"]
        bottom_right_lat = info["bottom_right_lat"]
        bottom_right_long = info["bottom_right_long"]

        line = [
            filename,
            top_left_lat,
            top_left_long,
            bottom_right_lat,
            bottom_right_long,
        ]
        lines.append(line)

    write_csv(output_path, lines, header)


def write_csv(csv_path: str, lines: list[list[Any]], header: list[str] = []):
    """Write the given lines to a CSV file."""
    print(f"Writing CSV file {csv_path}")
    with open(csv_path, "w", encoding="UTF8") as f:
        writer = csv.writer(f)
        if len(header) > 0:
            writer.writerow(header)

        for line in lines:
            writer.writerow(line)
