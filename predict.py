import sys
import csv
import os
import time
import socket
from datetime import datetime

import numpy as np
from PIL import Image
from tflite_runtime.interpreter import Interpreter


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "mobilenetv3_tb_pi.tflite"
RESULTS_FILE = "predictions.csv"
IMAGE_SIZE = (128, 128)
THRESHOLD = 0.5


# ============================================================
# GET IMAGE FROM COMMAND LINE
# ============================================================

if len(sys.argv) < 2:
    print()
    print("Usage:")
    print("python predict.py <image_name>")
    print()
    print("Example:")
    print("python predict.py test_xray.png")
    sys.exit(1)

IMAGE_PATH = sys.argv[1]


# ============================================================
# CHECK FILES EXIST
# ============================================================

if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model not found: {MODEL_PATH}")
    sys.exit(1)

if not os.path.exists(IMAGE_PATH):
    print(f"ERROR: Image not found: {IMAGE_PATH}")
    sys.exit(1)


# ============================================================
# LOAD TFLITE MODEL
# ============================================================

print()
print("Loading model...")

interpreter = Interpreter(
    model_path=MODEL_PATH
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Model loaded successfully!")


# ============================================================
# LOAD IMAGE
# ============================================================

print("Loading image...")

try:
    img = Image.open(
        IMAGE_PATH
    ).convert("RGB")

except Exception as e:
    print("ERROR: Could not load image.")
    print(e)
    sys.exit(1)


# ============================================================
# PREPROCESS IMAGE
# ============================================================

# Resize to the same dimensions used during training
img = img.resize(
    IMAGE_SIZE
)

# Convert image to NumPy array
img = np.array(img)

# Pillow loads RGB.
# Training used OpenCV which loads images as BGR,
# so convert RGB -> BGR to match training.
img = img[:, :, ::-1]

# Convert to float32 and normalize to 0-1
img = img.astype(
    np.float32
) / 255.0

# Add batch dimension:
# (128,128,3) -> (1,128,128,3)
img = np.expand_dims(
    img,
    axis=0
)

print("Input shape:", img.shape)


# ============================================================
# RUN INFERENCE
# ============================================================

print("Running prediction...")

interpreter.set_tensor(
    input_details[0]["index"],
    img
)

start_time = time.perf_counter()

interpreter.invoke()

end_time = time.perf_counter()

inference_ms = (
    end_time - start_time
) * 1000


# ============================================================
# GET MODEL OUTPUT
# ============================================================

output = interpreter.get_tensor(
    output_details[0]["index"]
)

probability = float(
    output[0][0]
)


# ============================================================
# DETERMINE PREDICTION
# ============================================================

if probability >= THRESHOLD:

    prediction = "Tuberculosis"
    confidence = probability

else:

    prediction = "Normal"
    confidence = 1.0 - probability


confidence_percent = confidence * 100


# ============================================================
# DISPLAY RESULT
# ============================================================

print()
print("========================================")
print("        TB SCREENING RESULT")
print("========================================")
print(f"Image:          {IMAGE_PATH}")
print(f"Prediction:     {prediction}")
print(f"TB probability: {probability:.4f}")
print(f"Confidence:     {confidence_percent:.2f}%")
print(f"Inference time: {inference_ms:.2f} ms")
print(f"Device:         {socket.gethostname()}")
print("========================================")


# ============================================================
# SAVE RESULT TO CSV
# ============================================================

file_exists = os.path.exists(
    RESULTS_FILE
)

with open(
    RESULTS_FILE,
    "a",
    newline=""
) as file:

    writer = csv.writer(file)

    # Create headings only the first time
    if not file_exists:

        writer.writerow([
            "timestamp",
            "image",
            "prediction",
            "tb_probability",
            "confidence_percent",
            "inference_ms",
            "device"
        ])

    writer.writerow([
        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        IMAGE_PATH,
        prediction,
        f"{probability:.4f}",
        f"{confidence_percent:.2f}",
        f"{inference_ms:.2f}",
        socket.gethostname()
    ])


print()
print(
    f"Result saved successfully to {RESULTS_FILE}"
)
print()
import numpy as np
from PIL import Image
from tflite_runtime.interpreter import Interpreter


MODEL_PATH = "mobilenetv3_tb_pi.tflite"


# ==========================================
# GET IMAGE FROM COMMAND LINE
# ==========================================

if len(sys.argv) < 2:
    print("Usage: python predict.py <image_name>")
    sys.exit(1)

IMAGE_PATH = sys.argv[1]


# ==========================================
# LOAD MODEL
# ==========================================

print("Loading model...")

interpreter = Interpreter(
    model_path=MODEL_PATH
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# ==========================================
# LOAD IMAGE
# ==========================================

print("Loading image...")

img = Image.open(
    IMAGE_PATH
).convert("RGB")


# ==========================================
# PREPROCESS IMAGE
# ==========================================

img = img.resize(
    (128, 128)
)

img = np.array(img)

# Match training preprocessing
img = img[:, :, ::-1]

img = img.astype(
    np.float32
) / 255.0

# Add batch dimension
img = np.expand_dims(
    img,
    axis=0
)


print("Input shape:", img.shape)
print("Running prediction...")


# ==========================================
# RUN MODEL
# ==========================================

interpreter.set_tensor(
    input_details[0]["index"],
    img
)

interpreter.invoke()


output = interpreter.get_tensor(
    output_details[0]["index"]
)


# ==========================================
# RESULT
# ==========================================

probability = float(
    output[0][0]
)


if probability >= 0.5:
    prediction = "Tuberculosis"
    confidence = probability
else:
    prediction = "Normal"
    confidence = 1 - probability


print()
print("==============================")
print("TB SCREENING RESULT")
print("==============================")
print("Prediction:", prediction)
print(f"TB probability: {probability:.4f}")
print(f"Confidence: {confidence*100:.2f}%")
print("==============================")
