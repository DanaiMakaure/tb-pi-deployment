import csv
import time
import statistics
from datetime import datetime

import numpy as np
from PIL import Image

from screening_engine import (
    IMAGE_SIZE,
    preprocess_image,
    predict,
)


# ============================================================
# CONFIGURATION
# ============================================================

WARMUP_RUNS = 10
MEASURED_RUNS = 100

CSV_FILE = "research_evidence/04_inference_benchmark.csv"
SUMMARY_FILE = "research_evidence/04_inference_summary.txt"


# ============================================================
# CREATE SAFE SYNTHETIC INPUT
#
# We benchmark model execution rather than diagnostic accuracy.
# TFLite uses the same fixed computation graph regardless of the
# image content, so no patient image is required for this timing
# experiment.
# ============================================================

synthetic_image = Image.fromarray(
    np.full(
        (
            IMAGE_SIZE[1],
            IMAGE_SIZE[0],
            3
        ),
        128,
        dtype=np.uint8
    ),
    mode="RGB"
)

model_input = preprocess_image(
    synthetic_image
)


print("======================================")
print("TB GUARD - INFERENCE BENCHMARK")
print("======================================")
print()
print(f"Image size: {IMAGE_SIZE}")
print(f"Warm-up runs: {WARMUP_RUNS}")
print(f"Measured runs: {MEASURED_RUNS}")
print()


# ============================================================
# WARM-UP
# ============================================================

print("Running warm-up...")

for _ in range(WARMUP_RUNS):
    predict(model_input)

print("Warm-up complete.")
print()


# ============================================================
# MEASURED RUNS
# ============================================================

results = []

print("Running measured inference...")

for run_number in range(1, MEASURED_RUNS + 1):

    call_start = time.perf_counter()

    result = predict(
        model_input
    )

    call_end = time.perf_counter()

    inference_ms = float(
        result["inference_ms"]
    )

    full_predict_call_ms = (
        call_end - call_start
    ) * 1000.0

    results.append({
        "run": run_number,
        "inference_ms": inference_ms,
        "full_predict_call_ms": full_predict_call_ms,
        "prediction": result["prediction"],
        "tb_probability": result["tb_probability"],
    })

    if run_number % 10 == 0:
        print(
            f"Completed "
            f"{run_number}/{MEASURED_RUNS}"
        )


# ============================================================
# SAVE RAW RESULTS
# ============================================================

with open(
    CSV_FILE,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "run",
            "inference_ms",
            "full_predict_call_ms",
            "prediction",
            "tb_probability",
        ]
    )

    writer.writeheader()
    writer.writerows(results)


# ============================================================
# STATISTICS
# ============================================================

times = [
    row["inference_ms"]
    for row in results
]

call_times = [
    row["full_predict_call_ms"]
    for row in results
]

mean_ms = statistics.mean(times)
median_ms = statistics.median(times)
std_ms = statistics.stdev(times)
min_ms = min(times)
max_ms = max(times)

percentile_95 = float(
    np.percentile(
        times,
        95
    )
)

mean_call_ms = statistics.mean(
    call_times
)

median_call_ms = statistics.median(
    call_times
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = f"""
======================================
TB GUARD - PI INFERENCE BENCHMARK
======================================

Test completed:
{datetime.now().isoformat()}

Image size:
{IMAGE_SIZE}

Warm-up runs:
{WARMUP_RUNS}

Measured runs:
{MEASURED_RUNS}

--------------------------------------
TFLITE MODEL INFERENCE ONLY
--------------------------------------

Mean inference time:
{mean_ms:.3f} ms

Median inference time:
{median_ms:.3f} ms

Standard deviation:
{std_ms:.3f} ms

Minimum inference time:
{min_ms:.3f} ms

Maximum inference time:
{max_ms:.3f} ms

95th percentile:
{percentile_95:.3f} ms

--------------------------------------
FULL predict() FUNCTION
--------------------------------------

Mean predict() call:
{mean_call_ms:.3f} ms

Median predict() call:
{median_call_ms:.3f} ms

--------------------------------------

Model:
mobilenetv3_tb_pi.tflite

Device:
Raspberry Pi 400
"""

with open(
    SUMMARY_FILE,
    "w"
) as f:
    f.write(summary)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print()
print(summary)

print("Raw timings saved to:")
print(CSV_FILE)

print()
print("Summary saved to:")
print(SUMMARY_FILE)
