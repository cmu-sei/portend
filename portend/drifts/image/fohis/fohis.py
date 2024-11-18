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

import copy

import numpy as np
from noise import pnoise3
from PIL import Image

from ..fohis import const as constant

const = constant.Const()

np.errstate(invalid="ignore", divide="ignore")

# atmosphere
const.VISIBILITY_RANGE_MOLECULE = 12  # m    12
const.VISIBILITY_RANGE_AEROSOL = 450  # m     450
const.ECM = (
    3.912 / const.VISIBILITY_RANGE_MOLECULE
)  # EXTINCTION_COEFFICIENT_MOLECULE /m
const.ECA = (
    3.912 / const.VISIBILITY_RANGE_AEROSOL
)  # EXTINCTION_COEFFICIENT_AEROSOL /m

const.FT = 70  # FOG_TOP m  31  70
const.HT = 34  # HAZE_TOP m  300    34

# camera
const.CAMERA_ALTITUDE = 1.8  # m fog 50   1.8
const.HORIZONTAL_ANGLE = 0  # °
const.CAMERA_VERTICAL_FOV = 64  # °


#
# Default parameters for fog drift
#
#
DEFAULT_PARAMS = {
    "vrm": 12.0,
    "vra": 450.0,
    "ft": 70.0,
    "ht": 34.0,
}


def get_image_info(src):
    im = Image.open(src)

    return im.info["dpi"]


def noise(Ip, depth):
    p1 = Image.new("L", (Ip.shape[1], Ip.shape[0]))
    p2 = Image.new("L", (Ip.shape[1], Ip.shape[0]))
    p3 = Image.new("L", (Ip.shape[1], Ip.shape[0]))

    scale = 1 / 130.0
    for y in range(Ip.shape[0]):
        for x in range(Ip.shape[1]):
            v = pnoise3(
                x * scale,
                y * scale,
                depth[y, x] * scale,
                octaves=1,
                persistence=0.5,
                lacunarity=2.0,
            )
            color = int((v + 1) * 128.0)
            p1.putpixel((x, y), color)

    scale = 1 / 60.0
    for y in range(Ip.shape[0]):
        for x in range(Ip.shape[1]):
            v = pnoise3(
                x * scale,
                y * scale,
                depth[y, x] * scale,
                octaves=1,
                persistence=0.5,
                lacunarity=2.0,
            )
            color = int((v + 0.5) * 128)
            p2.putpixel((x, y), color)

    scale = 1 / 10.0
    for y in range(Ip.shape[0]):
        for x in range(Ip.shape[1]):
            v = pnoise3(
                x * scale,
                y * scale,
                depth[y, x] * scale,
                octaves=1,
                persistence=0.5,
                lacunarity=2.0,
            )
            color = int((v + 1.2) * 128)
            p3.putpixel((x, y), color)

    perlin = (np.array(p1) + np.array(p2) / 2 + np.array(p3) / 4) / 3

    return perlin


#
#
#
def drift_images(img_list, params):
    const.VISIBILITY_RANGE_MOLECULE = float(
        params.get("vrm", DEFAULT_PARAMS["vrm"])
    )  # m    12
    const.VISIBILITY_RANGE_AEROSOL = float(
        params.get("vra", DEFAULT_PARAMS["vra"])
    )  # m     450
    const.ECM = (
        3.912 / const.VISIBILITY_RANGE_MOLECULE
    )  # EXTINCTION_COEFFICIENT_MOLECULE /m
    const.ECA = (
        3.912 / const.VISIBILITY_RANGE_AEROSOL
    )  # EXTINCTION_COEFFICIENT_AEROSOL /m
    const.FT = float(params.get("ft", DEFAULT_PARAMS["ft"]))  # m     450
    const.HT = float(params.get("ht", DEFAULT_PARAMS["ht"]))  # m     450

    out_img_list = []
    i = 0
    for img in img_list:
        print(f"Applying fohis drift to image {i}")
        out_img_list += [apply_fohis_drift(img)]
        i += 1

    return out_img_list


#
# Encode parameter settings into a string.
#
def encode_params(params=DEFAULT_PARAMS):
    vrm = float(params.get("vrm", DEFAULT_PARAMS["vrm"]))  # m    12
    vra = float(params.get("vra", DEFAULT_PARAMS["vra"]))  # m     450
    ft = float(params.get("ft", DEFAULT_PARAMS["ft"]))  # m     450
    ht = float(params.get("ht", DEFAULT_PARAMS["ht"]))  # m     450

    return "m%03da%03df%03dh%03d" % (
        int(vrm * 10),
        int(vra * 10),
        int(ft * 10),
        int(ht * 10),
    )


def _set_param(params, keys, value):
    if keys[0] not in params:
        return
    elif len(keys) > 1:
        _set_param(params[keys[0]], keys[1:], value)
    else:
        params[keys[0]] = value


#
# Decode url parameters.  Parameters are passed in a flat json structure with keys being the url parameters.
#
def decode_url_params(args):
    params = copy.deepcopy(DEFAULT_PARAMS)
    for k in args.keys():
        _set_param(params, k.split("."), args[k])
    return params


#
# Return Fohis drifted image.
#
def apply_fohis_drift(Ip):
    depth = np.ones(Ip.shape[0:2])
    depth[depth == 0] = 1  # the depth_min shouldn't be 0
    depth *= 3

    Imat = np.empty_like(Ip)
    result = np.empty_like(Ip)

    """
    elevation, distance, angle = tk.elevation_and_distance_estimation(img, depth,
                                                            const.CAMERA_VERTICAL_FOV,
                                                            const.HORIZONTAL_ANGLE,
                                                            const.CAMERA_ALTITUDE)
    """

    height, width = Ip.shape[:2]
    elevation = 100 * np.ones((height, width))
    distance = 100 * np.ones((height, width))
    _ = 90 * np.ones((height, width))

    if const.FT != 0:
        perlin = noise(Ip, depth)
        ECA = const.ECA
        # ECA = const.ECA * np.exp(-elevation/(const.FT+0.00001))
        c = 1 - elevation / (const.FT + 0.00001)
        c[c < 0] = 0

        if const.FT > const.HT:
            ECM = (const.ECM * c + (1 - c) * const.ECA) * (perlin / 255)
        else:
            ECM = (const.ECA * c + (1 - c) * const.ECM) * (perlin / 255)

    else:
        ECA = const.ECA
        # ECA = const.ECA * np.exp(-elevation/(const.FT+0.00001))
        ECM = const.ECM

    distance_through_fog = np.zeros_like(distance)
    distance_through_haze = np.zeros_like(distance)
    distance_through_haze_free = np.zeros_like(distance)

    if const.FT == 0:  # only haze: const.FT should be set to 0
        idx1 = elevation > const.HT
        idx2 = elevation <= const.HT

        if const.CAMERA_ALTITUDE <= const.HT:
            distance_through_haze[idx2] = distance[idx2]
            distance_through_haze_free[idx1] = (
                (elevation[idx1] - const.HT)
                * distance[idx1]
                / (elevation[idx1] - const.CAMERA_ALTITUDE)
            )

            distance_through_haze[idx1] = (
                distance[idx1] - distance_through_haze_free[idx1]
            )

        elif const.CAMERA_ALTITUDE > const.HT:
            distance_through_haze_free[idx1] = distance[idx1]
            distance_through_haze[idx2] = (
                (const.HT - elevation[idx2])
                * distance[idx2]
                / (const.CAMERA_ALTITUDE - elevation[idx2])
            )
            distance_through_haze_free[idx2] = (
                distance[idx2] - distance_through_fog[idx2]
            )

        Imat[:, :, 0] = Ip[:, :, 0] * np.exp(
            -ECA * distance_through_haze - ECM * distance_through_haze_free
        )
        Imat[:, :, 1] = Ip[:, :, 1] * np.exp(
            -ECA * distance_through_haze - ECM * distance_through_haze_free
        )
        Imat[:, :, 2] = Ip[:, :, 2] * np.exp(
            -ECA * distance_through_haze - ECM * distance_through_haze_free
        )
        Omat = 1 - np.exp(
            -ECA * distance_through_haze
            - const.ECM * distance_through_haze_free
        )

    elif const.FT < const.HT and const.FT != 0:
        idx1 = np.logical_and(const.HT > elevation, elevation > const.FT)
        idx2 = elevation <= const.FT
        idx3 = elevation >= const.HT
        if const.CAMERA_ALTITUDE <= const.FT:
            distance_through_fog[idx2] = distance[idx2]
            distance_through_haze[idx1] = (
                (elevation[idx1] - const.FT)
                * distance[idx1]
                / (elevation[idx1] - const.CAMERA_ALTITUDE)
            )

            distance_through_fog[idx1] = (
                distance[idx1] - distance_through_haze[idx1]
            )
            distance_through_fog[idx3] = (
                (const.FT - const.CAMERA_ALTITUDE)
                * distance[idx3]
                / (elevation[idx3] - const.CAMERA_ALTITUDE)
            )
            distance_through_haze[idx3] = (
                (const.HT - const.FT)
                * distance[idx3]
                / (elevation[idx3] - const.CAMERA_ALTITUDE)
            )
            distance_through_haze_free[idx3] = (
                distance[idx3]
                - distance_through_haze[idx3]
                - distance_through_fog[idx3]
            )

        elif const.CAMERA_ALTITUDE > const.HT:
            distance_through_haze_free[idx3] = distance[idx3]
            distance_through_haze[idx1] = (
                (const.FT - elevation[idx1])
                * distance_through_haze_free[idx1]
                / (const.CAMERA_ALTITUDE - const.HT)
            )
            distance_through_haze_free[idx1] = (
                distance[idx1] - distance_through_haze[idx1]
            )

            distance_through_fog[idx2] = (
                (const.FT - elevation[idx2])
                * distance[idx2]
                / (const.CAMERA_ALTITUDE - elevation[idx2])
            )
            distance_through_haze[idx2] = (
                (const.HT - const.FT)
                * distance
                / (const.CAMERA_ALTITUDE - elevation[idx2])
            )
            distance_through_haze_free[idx2] = (
                distance[idx2]
                - distance_through_haze[idx2]
                - distance_through_fog[idx2]
            )

        elif const.FT < const.CAMERA_ALTITUDE <= const.HT:
            distance_through_haze[idx1] = distance[idx1]
            distance_through_fog[idx2] = (
                (const.FT - elevation[idx2])
                * distance[idx2]
                / (const.CAMERA_ALTITUDE - elevation[idx2])
            )
            distance_through_haze[idx2] = (
                distance[idx2] - distance_through_fog[idx2]
            )
            distance_through_haze_free[idx3] = (
                (elevation[idx3] - const.HT)
                * distance[idx3]
                / (elevation[idx3] - const.CAMERA_ALTITUDE)
            )
            distance_through_haze[idx3] = (
                (const.HT - const.CAMERA_ALTITUDE)
                * distance[idx3]
                / (elevation[idx3] - const.CAMERA_ALTITUDE)
            )

        Imat[:, :, 0] = Ip[:, :, 0] * np.exp(
            -ECA * distance_through_haze - ECM * distance_through_fog
        )
        Imat[:, :, 1] = Ip[:, :, 1] * np.exp(
            -ECA * distance_through_haze - ECM * distance_through_fog
        )
        Imat[:, :, 2] = Ip[:, :, 2] * np.exp(
            -ECA * distance_through_haze - ECM * distance_through_fog
        )
        Omat = 1 - np.exp(
            -ECA * distance_through_haze - ECM * distance_through_fog
        )

    elif const.FT > const.HT:
        if const.CAMERA_ALTITUDE <= const.HT:
            idx1 = np.logical_and(const.FT > elevation, elevation > const.HT)
            idx2 = elevation <= const.HT
            idx3 = elevation >= const.FT

            distance_through_haze[idx2] = distance[idx2]
            distance_through_fog[idx1] = (
                (elevation[idx1] - const.HT)
                * distance[idx1]
                / (elevation[idx1] - const.CAMERA_ALTITUDE)
            )
            distance_through_haze[idx1] = (
                distance[idx1] - distance_through_fog[idx1]
            )
            distance_through_haze[idx3] = (
                (const.HT - const.CAMERA_ALTITUDE)
                * distance[idx3]
                / (elevation[idx3] - const.CAMERA_ALTITUDE)
            )
            distance_through_fog[idx3] = (
                (const.FT - const.HT)
                * distance[idx3]
                / (elevation[idx3] - const.CAMERA_ALTITUDE)
            )
            distance_through_haze_free[idx3] = (
                distance[idx3]
                - distance_through_haze[idx3]
                - distance_through_fog[idx3]
            )

        elif const.CAMERA_ALTITUDE > const.FT:
            idx1 = np.logical_and(const.HT > elevation, elevation > const.FT)
            idx2 = elevation <= const.FT
            idx3 = elevation >= const.HT

            distance_through_haze_free[idx3] = distance[idx3]
            distance_through_haze[idx1] = (
                (const.FT - elevation[idx1])
                * distance_through_haze_free[idx1]
                / (const.CAMERA_ALTITUDE - const.HT)
            )
            distance_through_haze_free[idx1] = (
                distance[idx1] - distance_through_haze[idx1]
            )
            distance_through_fog[idx2] = (
                (const.FT - elevation[idx2])
                * distance[idx2]
                / (const.CAMERA_ALTITUDE - elevation[idx2])
            )
            distance_through_haze[idx2] = (
                (const.HT - const.FT)
                * distance
                / (const.CAMERA_ALTITUDE - elevation[idx2])
            )
            distance_through_haze_free[idx2] = (
                distance[idx2]
                - distance_through_haze[idx2]
                - distance_through_fog[idx2]
            )

        elif const.HT < const.CAMERA_ALTITUDE <= const.FT:
            idx1 = np.logical_and(const.HT > elevation, elevation > const.FT)
            idx2 = elevation <= const.FT
            idx3 = elevation >= const.HT

            distance_through_haze[idx1] = distance[idx1]
            distance_through_fog[idx2] = (
                (const.FT - elevation[idx2])
                * distance[idx2]
                / (const.CAMERA_ALTITUDE - elevation[idx2])
            )
            distance_through_haze[idx2] = (
                distance[idx2] - distance_through_fog[idx2]
            )
            distance_through_haze_free[idx3] = (
                (elevation[idx3] - const.HT)
                * distance[idx3]
                / (elevation[idx3] - const.CAMERA_ALTITUDE)
            )
            distance_through_haze[idx3] = (
                (const.HT - const.CAMERA_ALTITUDE)
                * distance[idx3]
                / (elevation[idx3] - const.CAMERA_ALTITUDE)
            )

        Imat[:, :, 0] = Ip[:, :, 0] * np.exp(
            -ECA * distance_through_haze - ECM * distance_through_fog
        )
        Imat[:, :, 1] = Ip[:, :, 1] * np.exp(
            -ECA * distance_through_haze - ECM * distance_through_fog
        )
        Imat[:, :, 2] = Ip[:, :, 2] * np.exp(
            -ECA * distance_through_haze - ECM * distance_through_fog
        )
        Omat = 1 - np.exp(
            -ECA * distance_through_haze - ECM * distance_through_fog
        )

    Ial = np.empty_like(Ip)  # color of the fog/haze
    Ial[:, :, 0] = 225
    Ial[:, :, 1] = 225
    Ial[:, :, 2] = 201
    # Ial[:, :, 0] = 240
    # Ial[:, :, 1] = 240
    # Ial[:, :, 2] = 240

    result[:, :, 0] = Imat[:, :, 0] + Omat * Ial[:, :, 0]
    result[:, :, 1] = Imat[:, :, 1] + Omat * Ial[:, :, 1]
    result[:, :, 2] = Imat[:, :, 2] + Omat * Ial[:, :, 2]

    return result
