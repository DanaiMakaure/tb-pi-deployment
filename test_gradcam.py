import os
import sys
import math

import numpy as np

from PIL import Image

from tflite_runtime.interpreter import Interpreter


# ============================================================
# PATHS
# ============================================================

PREDICTION_MODEL = (
    "mobilenetv3_tb_pi.tflite"
)

FEATURE_MODEL = (
    "mobilenetv3_features_pi.tflite"
)

HEAD_WEIGHTS = (
    "mobilenetv3_gradcam_head.npz"
)

IMAGE_SIZE = (
    128,
    128
)

THRESHOLD = 0.5


# ============================================================
# COMMAND LINE
# ============================================================

if len(sys.argv) < 2:

    print()
    print(
        "Usage:"
    )

    print(
        "python test_gradcam.py image.png"
    )

    print()

    sys.exit(1)


IMAGE_PATH = sys.argv[1]


if not os.path.exists(
    IMAGE_PATH
):

    raise FileNotFoundError(
        f"Image not found: {IMAGE_PATH}"
    )


# ============================================================
# CHECK DEPLOYMENT FILES
# ============================================================

required_files = [

    PREDICTION_MODEL,

    FEATURE_MODEL,

    HEAD_WEIGHTS
]


for file_path in required_files:

    if not os.path.exists(
        file_path
    ):

        raise FileNotFoundError(
            f"Missing deployment file: {file_path}"
        )


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(
    image_path
):

    original = Image.open(
        image_path
    ).convert(
        "RGB"
    )


    resized = original.resize(
        IMAGE_SIZE,
        Image.Resampling.BILINEAR
    )


    image = np.array(
        resized
    )


    # --------------------------------------------------------
    # IMPORTANT
    #
    # Original training images were loaded with OpenCV,
    # therefore the model saw BGR channel ordering.
    #
    # Pillow loads RGB, so convert:
    #
    # RGB -> BGR
    # --------------------------------------------------------

    image = image[
        :,
        :,
        ::-1
    ]


    image = np.ascontiguousarray(
        image,
        dtype=np.float32
    )


    image = (
        image / 255.0
    )


    image = np.expand_dims(
        image,
        axis=0
    )


    return (
        original,
        image
    )


# ============================================================
# SIGMOID
# ============================================================

def sigmoid(
    value
):

    # Stable sigmoid

    if value >= 0:

        return (
            1.0
            /
            (
                1.0
                +
                math.exp(
                    -value
                )
            )
        )

    exponential = math.exp(
        value
    )

    return (
        exponential
        /
        (
            1.0
            +
            exponential
        )
    )


# ============================================================
# LOAD MAIN PREDICTION MODEL
# ============================================================

print()
print(
    "Loading prediction model..."
)


prediction_interpreter = Interpreter(
    model_path=PREDICTION_MODEL
)

prediction_interpreter.allocate_tensors()


prediction_input = (
    prediction_interpreter
    .get_input_details()
)

prediction_output = (
    prediction_interpreter
    .get_output_details()
)


print(
    "Prediction model loaded."
)


# ============================================================
# LOAD FEATURE MODEL
# ============================================================

print(
    "Loading Grad-CAM feature model..."
)


feature_interpreter = Interpreter(
    model_path=FEATURE_MODEL
)

feature_interpreter.allocate_tensors()


feature_input = (
    feature_interpreter
    .get_input_details()
)

feature_output = (
    feature_interpreter
    .get_output_details()
)


print(
    "Feature model loaded."
)


# ============================================================
# LOAD CLASSIFIER HEAD WEIGHTS
# ============================================================

print(
    "Loading classifier head..."
)


head = np.load(
    HEAD_WEIGHTS
)


W1 = head["W1"].astype(
    np.float32
)

b1 = head["b1"].astype(
    np.float32
)

W2 = head["W2"].astype(
    np.float32
)

b2 = head["b2"].astype(
    np.float32
)


print(
    "Classifier head loaded."
)


# ============================================================
# PREPARE IMAGE
# ============================================================

original_image, model_input = (
    preprocess_image(
        IMAGE_PATH
    )
)


print()
print(
    "Input image:",
    IMAGE_PATH
)

print(
    "Model input shape:",
    model_input.shape
)


# ============================================================
# MAIN MODEL PREDICTION
# ============================================================

prediction_interpreter.set_tensor(

    prediction_input[0][
        "index"
    ],

    model_input.astype(
        prediction_input[0][
            "dtype"
        ]
    )
)


prediction_interpreter.invoke()


main_output = (
    prediction_interpreter
    .get_tensor(
        prediction_output[0][
            "index"
        ]
    )
)


main_probability = float(
    main_output.reshape(
        -1
    )[0]
)


# ============================================================
# EXTRACT FEATURE MAPS
# ============================================================

feature_interpreter.set_tensor(

    feature_input[0][
        "index"
    ],

    model_input.astype(
        feature_input[0][
            "dtype"
        ]
    )
)


feature_interpreter.invoke()


feature_maps = (
    feature_interpreter
    .get_tensor(
        feature_output[0][
            "index"
        ]
    )
)


feature_maps = feature_maps[
    0
].astype(
    np.float32
)


print(
    "Feature map shape:",
    feature_maps.shape
)


# ============================================================
# RECONSTRUCT CLASSIFIER
# ============================================================

# Global Average Pooling
gap = np.mean(

    feature_maps,

    axis=(
        0,
        1
    )
)


# Dense(256)
hidden_pre_activation = (

    np.matmul(
        gap,
        W1
    )

    + b1
)


# ReLU
hidden = np.maximum(

    hidden_pre_activation,

    0.0
)


# Dense(1)
logit = float(

    np.matmul(
        hidden,
        W2
    ).reshape(
        -1
    )[0]

    + b2.reshape(
        -1
    )[0]
)


reconstructed_probability = (
    sigmoid(
        logit
    )
)


# ============================================================
# VERIFY MODELS MATCH
# ============================================================

difference = abs(

    main_probability

    -

    reconstructed_probability
)


print()
print(
    "============================================"
)

print(
    "MODEL CONSISTENCY CHECK"
)

print(
    "============================================"
)


print(
    f"Main TFLite probability: "
    f"{main_probability:.8f}"
)


print(
    f"Reconstructed probability: "
    f"{reconstructed_probability:.8f}"
)


print(
    f"Absolute difference: "
    f"{difference:.8f}"
)


if difference < 0.01:

    print(
        "Consistency: PASS"
    )

else:

    print(
        "Consistency: WARNING"
    )


# ============================================================
# DETERMINE CLASS
# ============================================================

if reconstructed_probability >= THRESHOLD:

    predicted_class = (
        "Tuberculosis"
    )

    confidence = (
        reconstructed_probability
    )

else:

    predicted_class = (
        "Normal"
    )

    confidence = (
        1.0
        -
        reconstructed_probability
    )


print()
print(
    "Prediction:",
    predicted_class
)


print(
    "TB probability:",
    f"{reconstructed_probability:.4f}"
)


print(
    "Confidence:",
    f"{confidence * 100:.2f}%"
)


# ============================================================
# TRUE GRAD-CAM
# ============================================================

height = feature_maps.shape[0]

width = feature_maps.shape[1]


# ------------------------------------------------------------
# Gradient through final sigmoid
#
# TB:
# score = p
#
# Normal:
# score = 1 - p
# ------------------------------------------------------------

sigmoid_gradient = (

    reconstructed_probability

    *

    (
        1.0
        -
        reconstructed_probability
    )
)


if predicted_class == "Tuberculosis":

    class_gradient = (
        sigmoid_gradient
    )

else:

    class_gradient = (
        -sigmoid_gradient
    )


# ------------------------------------------------------------
# Gradient through:
#
# Dense(256, ReLU)
# ->
# Dense(1)
# ------------------------------------------------------------

relu_mask = (

    hidden_pre_activation

    > 0
).astype(
    np.float32
)


output_weights = W2[
    :,
    0
]


gradient_gap = np.matmul(

    W1,

    (
        relu_mask
        *
        output_weights
    )
)


gradient_gap = (

    gradient_gap
    *
    class_gradient
)


# ------------------------------------------------------------
# GAP:
#
# gap = mean(feature maps)
#
# Therefore each spatial location contributes:
#
# 1 / (height * width)
# ------------------------------------------------------------

gradient_features = (

    gradient_gap

    /
    float(
        height
        *
        width
    )
)


# ------------------------------------------------------------
# Standard Grad-CAM channel weights
#
# alpha_k =
# mean spatial gradient for channel k
#
# Because GAP makes the spatial gradient constant,
# gradient_features already represents alpha.
# ------------------------------------------------------------

alpha = gradient_features


# ------------------------------------------------------------
# Weighted feature maps
# ------------------------------------------------------------

gradcam = np.sum(

    feature_maps

    *

    alpha.reshape(
        1,
        1,
        -1
    ),

    axis=-1
)


# ReLU
gradcam = np.maximum(

    gradcam,

    0.0
)


# ============================================================
# NORMALISE GRAD-CAM
# ============================================================

minimum = float(
    gradcam.min()
)

maximum = float(
    gradcam.max()
)


if maximum > minimum:

    gradcam = (

        gradcam
        -
        minimum

    ) / (

        maximum
        -
        minimum
    )

else:

    gradcam = np.zeros_like(
        gradcam
    )


print()
print(
    "Grad-CAM map:"
)

print(
    np.round(
        gradcam,
        3
    )
)


# ============================================================
# RESIZE HEATMAP
# ============================================================

heatmap_small = Image.fromarray(

    np.uint8(
        gradcam
        *
        255
    ),

    mode="L"
)


heatmap_large = heatmap_small.resize(

    original_image.size,

    Image.Resampling.BILINEAR
)


heat = np.array(

    heatmap_large,

    dtype=np.float32

) / 255.0


# ============================================================
# CREATE YELLOW -> ORANGE -> RED HEATMAP
# ============================================================

original_array = np.array(

    original_image,

    dtype=np.float32
)


colour_map = np.zeros_like(

    original_array,

    dtype=np.float32
)


# Red remains strong
colour_map[
    ...,
    0
] = 255.0


# Green:
#
# low activation  -> yellow
# high activation -> red

colour_map[
    ...,
    1
] = (

    255.0

    *

    (
        1.0
        -
        heat
    )
)


# Blue stays zero
colour_map[
    ...,
    2
] = 0.0


# ============================================================
# OVERLAY
# ============================================================

# Low activations are nearly transparent.
# Strong activations become much more visible.

alpha_overlay = (

    heat

    *
    0.65

)[
    ...,
    None
]


overlay_array = (

    original_array

    *
    (
        1.0
        -
        alpha_overlay
    )

    +

    colour_map

    *
    alpha_overlay
)


overlay_array = np.clip(

    overlay_array,

    0,

    255

).astype(
    np.uint8
)


overlay_image = Image.fromarray(
    overlay_array
)


# ============================================================
# SAVE OUTPUTS
# ============================================================

base_name = os.path.splitext(

    os.path.basename(
        IMAGE_PATH
    )

)[0]


heatmap_filename = (

    f"gradcam_heatmap_"
    f"{base_name}.png"
)


overlay_filename = (

    f"gradcam_overlay_"
    f"{base_name}.png"
)


heatmap_large.save(
    heatmap_filename
)


overlay_image.save(
    overlay_filename
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print(
    "============================================"
)

print(
    "GRAD-CAM COMPLETE"
)

print(
    "============================================"
)


print(
    "Class explained:",
    predicted_class
)


print(
    "Raw heatmap saved:",
    heatmap_filename
)


print(
    "Overlay saved:",
    overlay_filename
)


print()
print(
    "Yellow/orange/red areas represent "
    "regions that influenced the model's "
    f"{predicted_class} classification."
)

print()
