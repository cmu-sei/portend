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
import copy
import os

import cv2
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Add,
    Conv2D,
    Dense,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
    Permute,
    Reshape,
    UpSampling2D,
    multiply,
)

WEIGHT_FILE = "i2i.weights"

#
# Color of water in BGR format (backwards RGB)
#
WATER_COLOR = (0x19, 0x36, 0x4B)

#
# Image resolution and shapes for model input and output.
#
IMAGE_RESOLUTION = (256, 256)
INPUT_SHAPE = (256, 256, 3)
OUTPUT_SHAPE = (256, 256, 1)

#
# Maximum value of a pixel
#
PIXEL_MAX = 255

#
# Default parameters for flood drift
#
#  thresh: Threshold for water level (pixel value between 0 and 255)
#  trans: Transition range (in pixels values between 0 and 255)
#
DEFAULT_PARAMS = {
    "thresh": 4,
    "trans": 2,
    "color": {"r": 0x4B, "g": 0x36, "b": 0x19},
}

#
# Lazy creation of model
#
model = None


def squeeze_excite_block(tensor, ratio=16):
    init = tensor
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(
        filters // ratio,
        activation="relu",
        kernel_initializer="he_normal",
        use_bias=False,
    )(se)
    se = Dense(
        filters,
        activation="sigmoid",
        kernel_initializer="he_normal",
        use_bias=False,
    )(se)

    if K.image_data_format() == "channels_first":
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x


def F(x):
    return tf.math.log(x + tf.constant(0.5))


def create_loss(y_true, y_pred):
    y_diff = y_true - y_pred

    dx, dy = tf.image.image_gradients(y_diff)

    return 1.0 * K.mean((y_diff**2)) + 0.1 * (
        K.mean((dx**2)) + K.mean((dy**2))
    )


def create_model():
    KSIZE = 5

    input_layer = Input(shape=INPUT_SHAPE)

    E1a = Conv2D(
        32,
        (KSIZE, KSIZE),
        activation="relu",
        padding="same",
        input_shape=INPUT_SHAPE,
    )(input_layer)
    E1x = squeeze_excite_block(E1a)
    E1 = MaxPooling2D((2, 2), padding="same")(E1x)

    E2a = Conv2D(64, (KSIZE, KSIZE), activation="relu", padding="same")(E1)

    E2x = squeeze_excite_block(E2a)
    E2 = MaxPooling2D((2, 2), padding="same")(E2x)

    #
    # in: 64x64
    # out: 32x32
    #
    E3a = Conv2D(128, (KSIZE, KSIZE), activation="relu", padding="same")(E2)
    E3x = squeeze_excite_block(E3a)
    E3 = MaxPooling2D((2, 2), padding="same")(E3x)

    #
    # Middle layer
    # in: 32x32
    # out 32x32
    #
    E4a = Conv2D(256, (KSIZE, KSIZE), activation="relu", padding="same")(E3)
    E4x = squeeze_excite_block(E4a)
    E4 = Conv2D(128, (1, 1), activation="relu", padding="same")(E4x)

    D1a = UpSampling2D((2, 2))(E4)
    D1b = Conv2D(128, (KSIZE, KSIZE), activation="relu", padding="same")(D1a)
    D1 = Add()([D1b, E3x])

    D2a = UpSampling2D((2, 2))(D1)
    D2b = Conv2D(64, (KSIZE, KSIZE), activation="relu", padding="same")(D2a)
    D2 = Add()([D2b, E2x])

    D3a = UpSampling2D((2, 2))(D2)
    D3b = Conv2D(32, (KSIZE, KSIZE), activation="relu", padding="same")(D3a)
    D3 = Add()([D3b, E1x])

    # output_layer = Conv2D(1,(1,1),activation='relu',padding='same')(D3)

    R0 = Conv2D(128, (5, 5), activation="relu", padding="same")(D3)
    R1 = Conv2D(128, (5, 5), activation="relu", padding="same")(R0)
    # R2 = Conv2D(64,(5,5),activation='relu',padding='same')(R1)
    # R3 = Conv2D(32,(5,5),activation='relu',padding='same')(R0)

    output_layer = Conv2D(1, (1, 1), activation="relu", padding="same")(R1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model


def im2height(img):
    global model

    #
    # Save original image resolution
    #
    src_height = img.shape[0]
    src_width = img.shape[1]
    print(f"Src image resolution: height: {src_height}, width: {src_width}")

    #
    # Resize all images to model resolution and rescale pixel values to 0..1
    #
    img = cv2.resize(img, IMAGE_RESOLUTION) / float(PIXEL_MAX)

    #
    # Do the prediction
    #
    y = model.predict(np.array([img]))[0]

    #
    # Resize back to original resolution.  Note the resize also reshapes the height map into a flat 2d array.
    #
    y = cv2.resize(
        (y * PIXEL_MAX).astype(int),
        [src_width, src_height],
        interpolation=cv2.INTER_NEAREST,
    )
    print(f"Resized height map res: {y.shape}")

    return y


#
# It is assumed that 'base', and 'height' are three-element RGB arrays.  For the height
# map we assume that R=G=B.
#
def blend(base, height, transition, thresh, water_color):
    #
    # How much water at this location.
    #
    alpha = (height - thresh - transition) / (2 * transition)

    if alpha < 0:
        return water_color
    elif alpha > 1:
        return base
    else:
        return (water_color + alpha * (base - water_color)).astype(np.int64)


def blend_imgs(base_img, height_img, transition=2, thresh=4, color=WATER_COLOR):
    h = base_img.shape[0]
    w = base_img.shape[1]
    color = np.array(color)

    print(
        "* blending img({}) with height_map({})".format(
            base_img.shape, height_img.shape
        )
    )
    img = [
        [
            blend(base_img[r][c], height_img[r][c], transition, thresh, color)
            for c in range(w)
        ]
        for r in range(h)
    ]

    return np.array(img)


#
#
#
def drift_images(img_list, params):
    thresh = int(params.get("thresh", DEFAULT_PARAMS["thresh"]))
    trans = int(params.get("trans", DEFAULT_PARAMS["trans"]))
    color = params.get("color", DEFAULT_PARAMS["color"])
    color_red = int(color.get("r", DEFAULT_PARAMS["color"]["r"]))
    color_green = int(color.get("g", DEFAULT_PARAMS["color"]["g"]))
    color_blue = int(color.get("b", DEFAULT_PARAMS["color"]["b"]))

    flood_color = (color_blue, color_green, color_red)

    #
    # Generate the hight maps
    #
    height_list = [im2height(img) for img in img_list]

    #
    # Blend the original images with the height map to create the flooded tiles
    #
    img_list = [
        blend_imgs(
            img_list[i],
            height_list[i],
            transition=trans,
            thresh=thresh,
            color=flood_color,
        )
        for i in range(len(img_list))
    ]

    return img_list


#
# Encode parameter settings into a string.
#
def encode_params(params=DEFAULT_PARAMS):
    thresh = int(params.get("thresh", DEFAULT_PARAMS["thresh"]))
    trans = int(params.get("trans", DEFAULT_PARAMS["trans"]))
    color = params.get("color", DEFAULT_PARAMS["color"])
    color_red = int(color.get("r", DEFAULT_PARAMS["color"]["r"]))
    color_green = int(color.get("g", DEFAULT_PARAMS["color"]["g"]))
    color_blue = int(color.get("b", DEFAULT_PARAMS["color"]["b"]))

    return "t%da%dr%dg%db%d" % (
        thresh,
        trans,
        color_red,
        color_green,
        color_blue,
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
# Initial set-up
#
def setup():
    print("* [flood] initializing model")
    print(
        "* [flood] GPUs Available: ",
        len(tf.config.list_physical_devices("GPU")),
    )
    model = create_model()
    model.compile(optimizer="adam", loss=create_loss)

    curr_module_path = os.path.dirname(os.path.abspath(__file__))
    print(f"* [flood] curr module path :{curr_module_path}")
    print("* [flood] reading flood weights")
    full_weights_path = os.path.join(curr_module_path, WEIGHT_FILE)
    model.load_weights(full_weights_path).expect_partial()
    print("* [flood] model ready")


# Run setup when the module is loaded.
setup()
