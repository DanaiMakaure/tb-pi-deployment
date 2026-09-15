import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

from tflite_runtime.interpreter import Interpreter


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# MODEL FILES
# ============================================================

PREDICTION_MODEL_PATH = (
    BASE_DIR
    / "mobilenetv3_tb_pi.tflite"
)

FEATURE_MODEL_PATH = (
    BASE_DIR
    / "mobilenetv3_features_pi.tflite"
)

GRADCAM_HEAD_PATH = (
    BASE_DIR
    / "mobilenetv3_gradcam_head.npz"
)


# ============================================================
# SCREENING SETTINGS
# ============================================================

THRESHOLD = 0.5


# ============================================================
# GRAD-CAM CONSISTENCY SETTINGS
#
# <= 1%
#     normal pass
#
# > 1% and <= 5%
#     pass with warning IF both models predict same class
#
# > 5%
#     Grad-CAM rejected
#
# Different predicted classes
#     Grad-CAM rejected
#
# IMPORTANT:
# A Grad-CAM rejection does NOT invalidate the main screening.
# ============================================================

CONSISTENCY_WARNING_THRESHOLD = 0.01

CONSISTENCY_FAILURE_THRESHOLD = 0.05


# ============================================================
# PIL RESAMPLING COMPATIBILITY
# ============================================================

try:

    BILINEAR = (
        Image.Resampling.BILINEAR
    )

except AttributeError:

    BILINEAR = (
        Image.BILINEAR
    )


# ============================================================
# CHECK MODEL FILES
# ============================================================

required_files = [
    PREDICTION_MODEL_PATH,
    FEATURE_MODEL_PATH,
    GRADCAM_HEAD_PATH,
]


for required_file in required_files:

    if not required_file.exists():

        raise FileNotFoundError(
            "Required screening model file "
            "was not found: "
            f"{required_file}"
        )


# ============================================================
# LOAD MAIN PREDICTION MODEL
# ============================================================

print(
    "Loading MobileNetV3 prediction model..."
)


prediction_interpreter = Interpreter(
    model_path=str(
        PREDICTION_MODEL_PATH
    )
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
    "Prediction model ready."
)


# ============================================================
# LOAD GRAD-CAM FEATURE MODEL
# ============================================================

print(
    "Loading MobileNetV3 Grad-CAM feature model..."
)


feature_interpreter = Interpreter(
    model_path=str(
        FEATURE_MODEL_PATH
    )
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
    "Grad-CAM feature model ready."
)


# ============================================================
# DETERMINE INPUT IMAGE SIZE
#
# Expected:
#
# (1, 128, 128, 3)
# ============================================================

prediction_input_shape = (
    prediction_input[0][
        "shape"
    ]
)


if len(
    prediction_input_shape
) != 4:

    raise RuntimeError(
        "Unexpected prediction model input shape: "
        f"{prediction_input_shape}"
    )


INPUT_HEIGHT = int(
    prediction_input_shape[
        1
    ]
)


INPUT_WIDTH = int(
    prediction_input_shape[
        2
    ]
)


INPUT_CHANNELS = int(
    prediction_input_shape[
        3
    ]
)


if INPUT_CHANNELS != 3:

    raise RuntimeError(
        "The prediction model must use "
        "three image channels. "
        f"Found: {INPUT_CHANNELS}"
    )


IMAGE_SIZE = (
    INPUT_WIDTH,
    INPUT_HEIGHT,
)


# ============================================================
# VERIFY FEATURE MODEL INPUT
# ============================================================

feature_input_shape = (
    feature_input[0][
        "shape"
    ]
)


if tuple(
    feature_input_shape
) != tuple(
    prediction_input_shape
):

    raise RuntimeError(
        "Prediction model and Grad-CAM feature "
        "model use different input shapes. "
        f"Prediction: {prediction_input_shape}, "
        f"Feature: {feature_input_shape}"
    )


# ============================================================
# LOAD GRAD-CAM CLASSIFIER HEAD
# ============================================================

print(
    "Loading Grad-CAM classifier weights..."
)


head_data = np.load(
    str(
        GRADCAM_HEAD_PATH
    ),
    allow_pickle=False,
)


# ============================================================
# NPZ WEIGHT LOADER
#
# The exported model architecture is:
#
# feature maps
#     ↓
# GlobalAveragePooling
#     ↓
# Dense 256 + ReLU
#     ↓
# Dense 1 + Sigmoid
#
# W1 = (960, 256)
# b1 = (256,)
# W2 = (256, 1)
# b2 = (1,)
#
# We first try common key names.
# If those are different, we identify arrays by shape.
# ============================================================

def get_npz_array(
    possible_names,
    condition,
):

    for name in possible_names:

        if name in head_data.files:

            array = np.asarray(
                head_data[name],
                dtype=np.float32,
            )

            if condition(
                array
            ):

                return array


    for name in head_data.files:

        array = np.asarray(
            head_data[name],
            dtype=np.float32,
        )

        if condition(
            array
        ):

            return array


    return None


# ============================================================
# FIND W1
# ============================================================

W1 = get_npz_array(

    [
        "W1",
        "w1",
        "hidden_kernel",
        "dense_kernel",
        "kernel1",
    ],

    lambda array: (
        array.ndim == 2
        and
        array.shape[1] > 1
    ),
)


if W1 is None:

    raise RuntimeError(
        "Could not locate the hidden Dense "
        "weights in the Grad-CAM head."
    )


# ============================================================
# FIND b1
# ============================================================

hidden_units = int(
    W1.shape[
        1
    ]
)


b1 = get_npz_array(

    [
        "b1",
        "B1",
        "hidden_bias",
        "dense_bias",
        "bias1",
    ],

    lambda array: (
        array.ndim == 1
        and
        array.shape[0]
        ==
        hidden_units
    ),
)


if b1 is None:

    raise RuntimeError(
        "Could not locate the hidden Dense "
        "bias in the Grad-CAM head."
    )


# ============================================================
# FIND W2
# ============================================================

W2 = get_npz_array(

    [
        "W2",
        "w2",
        "output_kernel",
        "classifier_kernel",
        "kernel2",
    ],

    lambda array: (
        array.ndim == 2
        and
        array.shape[0]
        ==
        hidden_units
        and
        array.shape[1]
        ==
        1
    ),
)


if W2 is None:

    raise RuntimeError(
        "Could not locate the output Dense "
        "weights in the Grad-CAM head."
    )


# ============================================================
# FIND b2
# ============================================================

b2 = get_npz_array(

    [
        "b2",
        "B2",
        "output_bias",
        "classifier_bias",
        "bias2",
    ],

    lambda array: (
        array.ndim == 1
        and
        array.shape[0]
        ==
        1
    ),
)


if b2 is None:

    raise RuntimeError(
        "Could not locate the output Dense "
        "bias in the Grad-CAM head."
    )


# ============================================================
# VALIDATE CLASSIFIER HEAD
# ============================================================

if W1.shape[1] != b1.shape[0]:

    raise RuntimeError(
        "Hidden Dense weights and bias "
        "have incompatible shapes."
    )


if W2.shape[0] != W1.shape[1]:

    raise RuntimeError(
        "Hidden Dense output and final Dense "
        "input have incompatible shapes."
    )


if W2.shape[1] != 1:

    raise RuntimeError(
        "Final classifier must contain "
        "one sigmoid output."
    )


print(
    "Grad-CAM classifier weights ready."
)


# ============================================================
# ENGINE INFORMATION
# ============================================================

print()
print(
    "========================================"
)
print(
    "AI SCREENING ENGINE READY"
)
print(
    "========================================"
)
print(
    "Prediction model:",
    PREDICTION_MODEL_PATH.name,
)
print(
    "Feature model:",
    FEATURE_MODEL_PATH.name,
)
print(
    "Grad-CAM head:",
    GRADCAM_HEAD_PATH.name,
)
print(
    "========================================"
)
print()


# ============================================================
# TFLITE QUANTIZATION HELPERS
# ============================================================

def prepare_tensor(
    array,
    tensor_detail,
):

    target_dtype = (
        tensor_detail[
            "dtype"
        ]
    )


    if np.issubdtype(
        target_dtype,
        np.floating,
    ):

        return array.astype(
            target_dtype
        )


    scale, zero_point = (
        tensor_detail.get(
            "quantization",
            (
                0.0,
                0,
            )
        )
    )


    if (
        scale is None
        or
        float(scale) == 0.0
    ):

        return array.astype(
            target_dtype
        )


    quantized = np.round(
        (
            array
            /
            float(scale)
        )
        +
        int(zero_point)
    )


    limits = np.iinfo(
        target_dtype
    )


    quantized = np.clip(
        quantized,
        limits.min,
        limits.max,
    )


    return quantized.astype(
        target_dtype
    )


def read_tensor(
    interpreter,
    tensor_detail,
):

    output = interpreter.get_tensor(
        tensor_detail[
            "index"
        ]
    )


    output_dtype = (
        tensor_detail[
            "dtype"
        ]
    )


    if np.issubdtype(
        output_dtype,
        np.floating,
    ):

        return output.astype(
            np.float32
        )


    scale, zero_point = (
        tensor_detail.get(
            "quantization",
            (
                0.0,
                0,
            )
        )
    )


    if (
        scale is None
        or
        float(scale) == 0.0
    ):

        return output.astype(
            np.float32
        )


    return (
        (
            output.astype(
                np.float32
            )
            -
            float(
                zero_point
            )
        )
        *
        float(
            scale
        )
    )


# ============================================================
# STABLE SIGMOID
# ============================================================

def sigmoid(
    value
):

    value = float(
        value
    )


    if value >= 0.0:

        return float(
            1.0
            /
            (
                1.0
                +
                np.exp(
                    -value
                )
            )
        )


    exponential = np.exp(
        value
    )


    return float(
        exponential
        /
        (
            1.0
            +
            exponential
        )
    )


# ============================================================
# PREPROCESS IMAGE
#
# IMPORTANT:
#
# Training images were loaded using OpenCV.
# OpenCV uses BGR.
#
# Pillow uses RGB.
#
# Therefore:
#
# Pillow RGB
#     ↓
# resize
#     ↓
# RGB -> BGR
#     ↓
# float32
#     ↓
# divide by 255
#     ↓
# batch dimension
#
# BOTH model paths receive this exact same tensor.
# ============================================================

def preprocess_image(
    pil_image
):

    resized = (
        pil_image
        .convert(
            "RGB"
        )
        .resize(
            IMAGE_SIZE,
            BILINEAR,
        )
    )


    image = np.array(
        resized
    )


    # RGB -> BGR
    image = image[
        :,
        :,
        ::-1
    ]


    image = np.ascontiguousarray(
        image,
        dtype=np.float32,
    )


    # Normalize to 0-1
    image = (
        image
        /
        255.0
    )


    # Add batch dimension
    image = np.expand_dims(
        image,
        axis=0,
    )


    return image


# ============================================================
# MAIN TFLITE PREDICTION
#
# This model is authoritative.
# ============================================================

def predict(
    model_input
):

    prepared_input = prepare_tensor(
        model_input,
        prediction_input[0],
    )


    prediction_interpreter.set_tensor(
        prediction_input[0][
            "index"
        ],
        prepared_input,
    )


    start_time = time.perf_counter()


    prediction_interpreter.invoke()


    end_time = time.perf_counter()


    output = read_tensor(
        prediction_interpreter,
        prediction_output[0],
    )


    tb_probability = float(
        output.reshape(
            -1
        )[0]
    )


    # Protect against very small floating-point overflow.
    tb_probability = max(
        0.0,
        min(
            1.0,
            tb_probability,
        ),
    )


    inference_ms = (
        end_time
        -
        start_time
    ) * 1000.0


    if (
        tb_probability
        >=
        THRESHOLD
    ):

        prediction = (
            "Tuberculosis"
        )

        confidence = (
            tb_probability
        )


    else:

        prediction = (
            "Normal"
        )

        confidence = (
            1.0
            -
            tb_probability
        )


    return {
        "prediction":
            prediction,

        "tb_probability":
            float(
                tb_probability
            ),

        "confidence":
            float(
                confidence
            ),

        "inference_ms":
            float(
                inference_ms
            ),
    }


# ============================================================
# EXTRACT GRAD-CAM FEATURE MAPS
#
# Uses exactly the same model_input as predict().
# ============================================================

def extract_features(
    model_input
):

    prepared_input = prepare_tensor(
        model_input,
        feature_input[0],
    )


    feature_interpreter.set_tensor(
        feature_input[0][
            "index"
        ],
        prepared_input,
    )


    feature_interpreter.invoke()


    feature_maps = read_tensor(
        feature_interpreter,
        feature_output[0],
    )


    # Expected:
    #
    # (1, 4, 4, 960)
    # ->
    # (4, 4, 960)

    if (
        feature_maps.ndim
        ==
        4
    ):

        if (
            feature_maps.shape[
                0
            ]
            !=
            1
        ):

            raise RuntimeError(
                "Unexpected Grad-CAM feature "
                "batch dimension: "
                f"{feature_maps.shape}"
            )


        feature_maps = (
            feature_maps[
                0
            ]
            .astype(
                np.float32
            )
        )


    else:

        feature_maps = (
            feature_maps
            .astype(
                np.float32
            )
        )


    if feature_maps.ndim != 3:

        raise RuntimeError(
            "Unexpected Grad-CAM feature shape: "
            f"{feature_maps.shape}"
        )


    if (
        feature_maps.shape[
            -1
        ]
        !=
        W1.shape[
            0
        ]
    ):

        raise RuntimeError(
            "Feature model output does not match "
            "classifier head. "
            f"Feature channels: "
            f"{feature_maps.shape[-1]}, "
            f"classifier channels: "
            f"{W1.shape[0]}"
        )


    return feature_maps


# ============================================================
# RECONSTRUCT CLASSIFIER HEAD
#
# Feature maps:
#     (H, W, 960)
#
# GlobalAveragePooling:
#     (960,)
#
# Dense:
#     960 -> 256
#
# ReLU
#
# Dense:
#     256 -> 1
#
# Sigmoid
# ============================================================

def reconstruct_classifier(
    feature_maps
):

    # --------------------------------------------------------
    # GLOBAL AVERAGE POOLING
    # --------------------------------------------------------

    gap = np.mean(
        feature_maps,
        axis=(
            0,
            1,
        ),
        dtype=np.float32,
    )


    # --------------------------------------------------------
    # DENSE 256
    # --------------------------------------------------------

    hidden_pre = (
        np.matmul(
            gap,
            W1,
        )
        +
        b1
    )


    hidden_pre = (
        hidden_pre
        .astype(
            np.float32
        )
    )


    # --------------------------------------------------------
    # RELU
    # --------------------------------------------------------

    hidden = np.maximum(
        hidden_pre,
        0.0,
    )


    # --------------------------------------------------------
    # OUTPUT DENSE LOGIT
    # --------------------------------------------------------

    logit = float(
        np.matmul(
            hidden,
            W2,
        )
        .reshape(
            -1
        )[0]
        +
        b2.reshape(
            -1
        )[0]
    )


    # --------------------------------------------------------
    # SIGMOID
    # --------------------------------------------------------

    probability = sigmoid(
        logit
    )


    return (
        float(
            probability
        ),
        hidden_pre,
        float(
            logit
        ),
    )


# ============================================================
# MODEL CONSISTENCY CHECK
#
# IMPORTANT:
#
# This function still raises RuntimeError when Grad-CAM is
# unsafe.
#
# analyse_xray() catches that error so the MAIN screening
# prediction survives.
# ============================================================

def check_model_consistency(
    main_probability,
    reconstructed_probability,
):

    main_probability = float(
        main_probability
    )


    reconstructed_probability = float(
        reconstructed_probability
    )


    probability_difference = abs(
        main_probability
        -
        reconstructed_probability
    )


    # --------------------------------------------------------
    # MAIN MODEL CLASS
    # --------------------------------------------------------

    main_class = (

        "Tuberculosis"

        if (
            main_probability
            >=
            THRESHOLD
        )

        else

        "Normal"
    )


    # --------------------------------------------------------
    # GRAD-CAM MODEL CLASS
    # --------------------------------------------------------

    gradcam_class = (

        "Tuberculosis"

        if (
            reconstructed_probability
            >=
            THRESHOLD
        )

        else

        "Normal"
    )


    print()
    print(
        "========================================"
    )
    print(
        "GRAD-CAM CONSISTENCY CHECK"
    )
    print(
        "========================================"
    )

    print(
        "Main model probability:",
        f"{main_probability:.8f}",
    )

    print(
        "Grad-CAM probability:",
        f"{reconstructed_probability:.8f}",
    )

    print(
        "Absolute difference:",
        f"{probability_difference:.8f}",
    )

    print(
        "Main model class:",
        main_class,
    )

    print(
        "Grad-CAM class:",
        gradcam_class,
    )


    # --------------------------------------------------------
    # CLASS DISAGREEMENT
    # --------------------------------------------------------

    if (
        main_class
        !=
        gradcam_class
    ):

        print(
            "Consistency: FAILED"
        )

        print(
            "Reason: predicted classes disagree."
        )

        print(
            "========================================"
        )
        print()


        raise RuntimeError(

            "Prediction model and Grad-CAM model "
            "disagree on the predicted class. "

            f"Prediction model: "
            f"{main_class}. "

            f"Grad-CAM model: "
            f"{gradcam_class}."
        )


    # --------------------------------------------------------
    # LARGE PROBABILITY DIFFERENCE
    # --------------------------------------------------------

    if (
        probability_difference
        >
        CONSISTENCY_FAILURE_THRESHOLD
    ):

        print(
            "Consistency: FAILED"
        )

        print(
            "Difference exceeds safety threshold."
        )

        print(
            "========================================"
        )
        print()


        raise RuntimeError(

            "Prediction model and Grad-CAM model "
            "differ too much. "

            f"Difference: "
            f"{probability_difference:.6f}"
        )


    # --------------------------------------------------------
    # WARNING RANGE
    # --------------------------------------------------------

    if (
        probability_difference
        >
        CONSISTENCY_WARNING_THRESHOLD
    ):

        consistency_status = (
            "PASS WITH WARNING"
        )


        print(
            "Consistency:",
            consistency_status,
        )

        print(
            "Both model paths agree on the predicted "
            "class, but their probability values differ "
            "slightly."
        )


    else:

        consistency_status = (
            "PASS"
        )


        print(
            "Consistency:",
            consistency_status,
        )


    print(
        "========================================"
    )
    print()


    return {
        "difference":
            float(
                probability_difference
            ),

        "main_class":
            main_class,

        "gradcam_class":
            gradcam_class,

        "status":
            consistency_status,
    }


# ============================================================
# HEATMAP COLOUR MAP
#
# Small NumPy implementation so we do not require matplotlib
# or OpenCV on the Raspberry Pi.
# ============================================================

def colourise_heatmap(
    heatmap
):

    heatmap = np.clip(
        heatmap,
        0.0,
        1.0,
    )


    # Approximate "jet" colour map.
    red = np.clip(
        1.5
        -
        np.abs(
            4.0
            *
            heatmap
            -
            3.0
        ),
        0.0,
        1.0,
    )


    green = np.clip(
        1.5
        -
        np.abs(
            4.0
            *
            heatmap
            -
            2.0
        ),
        0.0,
        1.0,
    )


    blue = np.clip(
        1.5
        -
        np.abs(
            4.0
            *
            heatmap
            -
            1.0
        ),
        0.0,
        1.0,
    )


    colour = np.stack(
        [
            red,
            green,
            blue,
        ],
        axis=-1,
    )


    return (
        colour
        *
        255.0
    ).astype(
        np.uint8
    )


# ============================================================
# GENERATE TRUE GRAD-CAM
#
# IMPORTANT:
#
# We calculate the gradient from the classifier LOGIT.
#
# This avoids sigmoid gradient saturation when:
#
# probability ≈ 1
#
# or:
#
# probability ≈ 0
# ============================================================

def generate_gradcam(
    original_image,
    feature_maps,
    hidden_pre,
    prediction,
):

    height = int(
        feature_maps.shape[
            0
        ]
    )


    width = int(
        feature_maps.shape[
            1
        ]
    )


    # --------------------------------------------------------
    # RELU DERIVATIVE
    # --------------------------------------------------------

    relu_mask = (
        hidden_pre
        >
        0.0
    ).astype(
        np.float32
    )


    # --------------------------------------------------------
    # FINAL DENSE OUTPUT WEIGHTS
    #
    # (256, 1)
    # ->
    # (256,)
    # --------------------------------------------------------

    output_weights = (
        W2[
            :,
            0
        ]
    )


    # --------------------------------------------------------
    # GRADIENT OF LOGIT WITH RESPECT TO GAP
    #
    # W1:
    #     (960,256)
    #
    # relu_mask * output_weights:
    #     (256,)
    #
    # gradient_gap:
    #     (960,)
    # --------------------------------------------------------

    gradient_gap = np.matmul(

        W1,

        (
            relu_mask
            *
            output_weights
        )
    )


    # --------------------------------------------------------
    # CLASS-SPECIFIC SCORE
    #
    # TB:
    #     +logit
    #
    # Normal:
    #     -logit
    # --------------------------------------------------------

    if (
        prediction
        ==
        "Normal"
    ):

        gradient_gap = (
            -gradient_gap
        )


    # --------------------------------------------------------
    # GRADIENT THROUGH GLOBAL AVERAGE POOLING
    #
    # GAP averages every spatial location.
    #
    # dGAP/dFeature =
    #
    #     1 / (height * width)
    #
    # --------------------------------------------------------

    spatial_count = float(
        height
        *
        width
    )


    channel_weights = (
        gradient_gap
        /
        spatial_count
    )


    # --------------------------------------------------------
    # WEIGHT FEATURE MAP CHANNELS
    # --------------------------------------------------------

    heatmap = np.sum(

        feature_maps
        *
        channel_weights[
            np.newaxis,
            np.newaxis,
            :
        ],

        axis=-1,
    )


    # --------------------------------------------------------
    # GRAD-CAM RELU
    # --------------------------------------------------------

    heatmap = np.maximum(
        heatmap,
        0.0,
    )


    maximum = float(
        np.max(
            heatmap
        )
    )


    if (
        not np.isfinite(
            maximum
        )
        or
        maximum
        <=
        1e-12
    ):

        raise RuntimeError(
            "Grad-CAM produced no usable positive "
            "activation for this image."
        )


    # --------------------------------------------------------
    # NORMALISE TO 0-1
    # --------------------------------------------------------

    heatmap = (
        heatmap
        /
        maximum
    )


    heatmap = np.clip(
        heatmap,
        0.0,
        1.0,
    )


    # --------------------------------------------------------
    # RESIZE HEATMAP TO ORIGINAL X-RAY SIZE
    # --------------------------------------------------------

    heatmap_gray = Image.fromarray(
        (
            heatmap
            *
            255.0
        ).astype(
            np.uint8
        ),
        mode="L",
    )


    heatmap_gray = heatmap_gray.resize(
        original_image.size,
        BILINEAR,
    )


    resized_heatmap = (
        np.asarray(
            heatmap_gray,
            dtype=np.float32,
        )
        /
        255.0
    )


    # --------------------------------------------------------
    # CREATE COLOURED HEATMAP
    # --------------------------------------------------------

    coloured_array = colourise_heatmap(
        resized_heatmap
    )


    heatmap_image = Image.fromarray(
        coloured_array,
        mode="RGB",
    )


    # --------------------------------------------------------
    # CREATE OVERLAY
    # --------------------------------------------------------

    original_rgb = (
        original_image
        .convert(
            "RGB"
        )
    )


    original_array = np.asarray(
        original_rgb,
        dtype=np.float32,
    )


    colour_array = np.asarray(
        heatmap_image,
        dtype=np.float32,
    )


    # Heatmap is stronger only where activation exists.
    activation = (
        resized_heatmap[
            :,
            :,
            np.newaxis
        ]
    )


    alpha = (
        0.45
        *
        activation
    )


    overlay_array = (
        original_array
        *
        (
            1.0
            -
            alpha
        )
        +
        colour_array
        *
        alpha
    )


    overlay_array = np.clip(
        overlay_array,
        0.0,
        255.0,
    ).astype(
        np.uint8
    )


    overlay_image = Image.fromarray(
        overlay_array,
        mode="RGB",
    )


    return (
        heatmap_image,
        overlay_image,
    )


# ============================================================
# REMOVE OLD EXPLANATION FILE
# ============================================================

def remove_explanation_file(
    path
):

    if not path:
        return


    try:

        path = Path(
            path
        )


        if (
            path.exists()
            and
            path.is_file()
        ):

            path.unlink()


    except Exception as exc:

        print(
            "Could not remove stale "
            "Grad-CAM file:",
            path,
            exc,
        )


# ============================================================
# COMPLETE X-RAY ANALYSIS
#
# IMPORTANT ARCHITECTURE:
#
# Main prediction model
#     |
#     | authoritative result
#     ↓
# prediction
#
# Grad-CAM path
#     |
#     ↓
# feature model
#     |
#     ↓
# reconstructed classifier
#     |
#     ↓
# consistency check
#     |
#     ├── safe
#     │     ↓
#     │   generate Grad-CAM
#     │
#     └── mismatch
#           ↓
#         suppress Grad-CAM
#
# A Grad-CAM failure NEVER changes the main prediction.
# ============================================================

def analyse_xray(
    image_path,
    overlay_path,
    heatmap_path=None,
):

    # --------------------------------------------------------
    # LOAD X-RAY
    # --------------------------------------------------------

    original_image = (
        Image.open(
            image_path
        )
        .convert(
            "RGB"
        )
    )


    # --------------------------------------------------------
    # SINGLE SHARED PREPROCESSING PATH
    # --------------------------------------------------------

    model_input = preprocess_image(
        original_image
    )


    # ========================================================
    # MAIN SCREENING MODEL
    #
    # If this fails, the screening itself genuinely failed.
    # ========================================================

    result = predict(
        model_input
    )


    # ========================================================
    # DEFAULT EXPLANATION STATE
    # ========================================================

    result[
        "gradcam_available"
    ] = False


    result[
        "gradcam_reason"
    ] = None


    result[
        "reconstructed_probability"
    ] = None


    result[
        "consistency_difference"
    ] = None


    result[
        "consistency_status"
    ] = "NOT CHECKED"


    result[
        "consistency_main_class"
    ] = result[
        "prediction"
    ]


    result[
        "consistency_gradcam_class"
    ] = None


    result[
        "logit"
    ] = None


    result[
        "feature_shape"
    ] = None


    # --------------------------------------------------------
    # Ensure an older explanation cannot accidentally survive
    # if the new consistency check fails.
    # --------------------------------------------------------

    remove_explanation_file(
        overlay_path
    )


    if heatmap_path:

        remove_explanation_file(
            heatmap_path
        )


    # ========================================================
    # EXPLANATION PIPELINE
    #
    # Anything failing below here affects Grad-CAM only.
    # ========================================================

    try:

        # ----------------------------------------------------
        # EXTRACT FEATURE MAPS
        # ----------------------------------------------------

        feature_maps = extract_features(
            model_input
        )


        result[
            "feature_shape"
        ] = tuple(
            int(value)
            for value
            in feature_maps.shape
        )


        # ----------------------------------------------------
        # RECONSTRUCT CLASSIFIER
        # ----------------------------------------------------

        (
            reconstructed_probability,
            hidden_pre,
            logit,

        ) = reconstruct_classifier(
            feature_maps
        )


        reconstructed_probability = float(
            reconstructed_probability
        )


        logit = float(
            logit
        )


        result[
            "reconstructed_probability"
        ] = reconstructed_probability


        result[
            "logit"
        ] = logit


        # ----------------------------------------------------
        # SAVE DIFFERENCE EVEN IF CONSISTENCY FAILS
        # ----------------------------------------------------

        probability_difference = abs(

            float(
                result[
                    "tb_probability"
                ]
            )

            -

            reconstructed_probability
        )


        result[
            "consistency_difference"
        ] = float(
            probability_difference
        )


        gradcam_class = (

            "Tuberculosis"

            if (
                reconstructed_probability
                >=
                THRESHOLD
            )

            else

            "Normal"
        )


        result[
            "consistency_gradcam_class"
        ] = gradcam_class


        # ====================================================
        # CONSISTENCY CHECK
        # ====================================================

        try:

            consistency = (
                check_model_consistency(

                    result[
                        "tb_probability"
                    ],

                    reconstructed_probability,
                )
            )


        except RuntimeError as consistency_error:

            # ------------------------------------------------
            # MAIN MODEL RESULT REMAINS VALID.
            #
            # Only the explanation is rejected.
            # ------------------------------------------------

            result[
                "consistency_status"
            ] = "FAILED"


            result[
                "gradcam_available"
            ] = False


            result[
                "gradcam_reason"
            ] = str(
                consistency_error
            )


            remove_explanation_file(
                overlay_path
            )


            if heatmap_path:

                remove_explanation_file(
                    heatmap_path
                )


            print()
            print(
                "========================================"
            )
            print(
                "GRAD-CAM NOT AVAILABLE"
            )
            print(
                "========================================"
            )

            print(
                "Main screening prediction:",
                result[
                    "prediction"
                ],
            )

            print(
                "Main model probability:",
                f"{result['tb_probability']:.8f}",
            )

            print(
                "Grad-CAM probability:",
                f"{reconstructed_probability:.8f}",
            )

            print(
                "Absolute difference:",
                f"{probability_difference:.8f}",
            )

            print(
                "Reason:",
                result[
                    "gradcam_reason"
                ],
            )

            print(
                "Main screening result preserved."
            )

            print(
                "Grad-CAM explanation suppressed."
            )

            print(
                "========================================"
            )
            print()


            return result


        # ====================================================
        # CONSISTENCY PASSED
        # ====================================================

        result[
            "consistency_difference"
        ] = float(
            consistency[
                "difference"
            ]
        )


        result[
            "consistency_status"
        ] = consistency[
            "status"
        ]


        result[
            "consistency_main_class"
        ] = consistency[
            "main_class"
        ]


        result[
            "consistency_gradcam_class"
        ] = consistency[
            "gradcam_class"
        ]


        # ====================================================
        # GENERATE GRAD-CAM
        # ====================================================

        (
            heatmap_image,
            overlay_image,

        ) = generate_gradcam(

            original_image,

            feature_maps,

            hidden_pre,

            result[
                "prediction"
            ],
        )


        # ----------------------------------------------------
        # CREATE OVERLAY DIRECTORY
        # ----------------------------------------------------

        overlay_directory = (
            os.path.dirname(
                overlay_path
            )
        )


        if overlay_directory:

            os.makedirs(
                overlay_directory,
                exist_ok=True,
            )


        # ----------------------------------------------------
        # SAVE OVERLAY
        # ----------------------------------------------------

        overlay_image.save(
            overlay_path,
            format="PNG",
        )


        # ----------------------------------------------------
        # OPTIONAL RAW HEATMAP
        # ----------------------------------------------------

        if heatmap_path:

            heatmap_directory = (
                os.path.dirname(
                    heatmap_path
                )
            )


            if heatmap_directory:

                os.makedirs(
                    heatmap_directory,
                    exist_ok=True,
                )


            heatmap_image.save(
                heatmap_path,
                format="PNG",
            )


        # ----------------------------------------------------
        # GRAD-CAM SUCCESS
        # ----------------------------------------------------

        result[
            "gradcam_available"
        ] = True


        result[
            "gradcam_reason"
        ] = None


        return result


    # ========================================================
    # OTHER GRAD-CAM FAILURE
    #
    # Examples:
    #
    # - feature-model error
    # - classifier-head error
    # - zero Grad-CAM activation
    # - image-writing error
    #
    # Main prediction still survives.
    # ========================================================

    except Exception as gradcam_error:

        remove_explanation_file(
            overlay_path
        )


        if heatmap_path:

            remove_explanation_file(
                heatmap_path
            )


        result[
            "gradcam_available"
        ] = False


        result[
            "gradcam_reason"
        ] = (
            "Grad-CAM explanation could not "
            "be generated. "
            +
            str(
                gradcam_error
            )
        )


        if (
            result[
                "consistency_status"
            ]
            ==
            "NOT CHECKED"
        ):

            result[
                "consistency_status"
            ] = "GRAD-CAM ERROR"


        print()
        print(
            "========================================"
        )
        print(
            "GRAD-CAM PIPELINE ERROR"
        )
        print(
            "========================================"
        )

        print(
            "Screening prediction preserved:",
            result[
                "prediction"
            ],
        )

        print(
            "TB probability:",
            f"{result['tb_probability']:.8f}",
        )

        print(
            "Grad-CAM available: False"
        )

        print(
            "Reason:",
            result[
                "gradcam_reason"
            ],
        )

        print(
            "========================================"
        )
        print()


        return result
