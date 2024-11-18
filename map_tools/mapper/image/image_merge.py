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

import cv2
import numpy as np
import numpy.typing as npt

TILE_WIDTH = 256
TILE_HEIGHT = 256
TILE_CHANNEL = 3


def speed_block_2D(chops: npt.NDArray[Any]):
    H = np.cumsum([x[0].shape[0] for x in chops])
    W = np.cumsum([x.shape[1] for x in chops[0]])
    D = chops[0][0]
    recon = np.empty((H[-1], W[-1], D.shape[2]), D.dtype)
    for rd, rs in zip(np.split(recon, H[:-1], 0), chops):
        for d, s in zip(np.split(rd, W[:-1], 1), rs):
            d[...] = s
    return recon


def combine_images(image_names: list[list[str]], final_name: str):
    """Gets a list of image paths from disk, creates a new combined one as a result."""
    images = np.zeros(
        shape=[
            len(image_names),
            len(image_names[0]),
            TILE_WIDTH,
            TILE_HEIGHT,
            TILE_CHANNEL,
        ]
    )
    for x, image_row in enumerate(image_names):
        for y, image in enumerate(image_row):
            images[x][y] = cv2.imread(image)

    # Combine and store this image.
    combined_image = speed_block_2D(images)
    cv2.imwrite(final_name, combined_image)
