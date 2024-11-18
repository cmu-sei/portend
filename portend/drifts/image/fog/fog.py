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

# mypy: ignore-errors
from __future__ import annotations

import copy
import typing
from typing import Any, Dict

import cv2
import numpy as np

#
# Default parameters for fog drift
#
#  gray: Grayscale level for fog (0=black, 1=white)
#  blend: Mix between greyscale color and image base (0=pure-image, 1=pure-gray)
#  noise: Pixel noise mixing (1.0=full deflection)
#  blur.radius: Radius of blur (0 or 1 for no blur)
#  blur.alpha: Mixing parameter for blue (1=full blur, 0=no blur)
#
DEFAULT_PARAMS = {
    "gray": 0.3,
    "blend": 0.5,
    "noise": 0.0,
    "blur": {"radius": 0, "alpha": 1},
}


#
#
#
def drift_images(img_list, params: dict[str, Any]):
    print(DEFAULT_PARAMS)
    gray = float(params.get("gray", DEFAULT_PARAMS["gray"]))
    blend = float(params.get("blend", DEFAULT_PARAMS["blend"]))
    noise = float(params.get("noise", DEFAULT_PARAMS["noise"]))
    blur = typing.cast(
        Dict[str, Any], params.get("blur", DEFAULT_PARAMS["blur"])
    )

    gray_mat = np.full(img_list[0].shape, gray * 255)
    img_list = [
        np.clip(
            gray_mat * blend
            + img * (1 - blend)
            + (np.random.random(img.shape) * 2 - 1) * 255 * noise,
            0,
            255,
        )
        for img in img_list
    ]

    if blur:
        blur_radius = int(blur.get("radius", DEFAULT_PARAMS["blur"]["radius"]))
        blur_alpha = float(blur.get("alpha", DEFAULT_PARAMS["blur"]["alpha"]))
        k = (blur_radius - 1) * 2 + 1
        if blur_radius > 1:
            kernel = np.ones((k, k)) * blur_alpha
            kernel[blur_radius - 1, blur_radius - 1] = 1
            kernel = kernel / np.sum(kernel)
            img_list = [cv2.filter2D(img, -1, kernel) for img in img_list]

    return img_list


#
# Encode parameter settings into a string.
#
def encode_params(params=DEFAULT_PARAMS):
    gray = float(params.get("gray", DEFAULT_PARAMS["gray"]))
    blend = float(params.get("blend", DEFAULT_PARAMS["blend"]))
    noise = float(params.get("noise", DEFAULT_PARAMS["noise"]))
    blur = params.get("blur", DEFAULT_PARAMS["blur"])
    if blur:
        blur_radius = int(blur.get("radius", DEFAULT_PARAMS["blur"]["radius"]))
        blur_alpha = float(blur.get("alpha", DEFAULT_PARAMS["blur"]["alpha"]))
    else:
        blur_radius = 1

    if blur_radius <= 1:
        blur_alpha = 0

    return "g%03db%03dr%03da%03dn%03d" % (
        int(gray * 100),
        int(blend * 100),
        int(blur_radius),
        int(blur_alpha),
        int(noise * 100),
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
