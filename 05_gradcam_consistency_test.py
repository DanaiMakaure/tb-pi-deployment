import os
import csv
import statistics
from datetime import datetime

from PIL import Image

from screening_engine import (
    preprocess_image,
    predict,
    extract_features,
    reconstruct_classifier,
    check_model_consistency,
    CONSISTENCY_WARNING_THRESHOLD,
    CONSISTENCY_FAILURE_THRESHOLD,
)

TEST_FOLDER = "gradcam_test_images"

CSV_FILE = "research_evidence/05_gradcam_consistency.csv"
SUMMARY_FILE = "research_evidence/05_gradcam_summary.txt"

VALID_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
)


# ============================================================
# FIND TEST IMAGES
# ============================================================

image_files = sorted([
    filename
    for filename in os.listdir(TEST_FOLDER)
    if filename.lower().endswith(VALID_EXTENSIONS)
])

if not image_files:
    raise RuntimeError(
        f"No test images found in {TEST_FOLDER}"
    )


print("======================================")
print("TB GUARD - GRAD-CAM CONSISTENCY TEST")
print("======================================")
print()

print(
    f"Images found: {len(image_files)}"
)

print(
    "Warning threshold:",
    CONSISTENCY_WARNING_THRESHOLD
)

print(
    "Failure threshold:",
    CONSISTENCY_FAILURE_THRESHOLD
)

print()


# ============================================================
# RUN TEST
# ============================================================

rows = []

for index, filename in enumerate(
    image_files,
    start=1
):

    anonymous_id = (
        f"image_{index:03d}"
    )

    path = os.path.join(
        TEST_FOLDER,
        filename
    )

    print()
    print(
        "--------------------------------------"
    )

    print(
        f"Testing {anonymous_id}"
    )

    try:

        with Image.open(path) as image:

            model_input = preprocess_image(
                image
            )


        # --------------------------------------
        # AUTHORITATIVE PREDICTION MODEL
        # --------------------------------------

        main_result = predict(
            model_input
        )

        main_probability = float(
            main_result[
                "tb_probability"
            ]
        )


        # --------------------------------------
        # GRAD-CAM FEATURE PATH
        # --------------------------------------

        feature_maps = extract_features(
            model_input
        )

        (
            reconstructed_probability,
            _hidden_pre,
            _logit,
        ) = reconstruct_classifier(
            feature_maps
        )


        difference = abs(
            main_probability
            -
            reconstructed_probability
        )


        # --------------------------------------
        # EXISTING SAFETY CHECK
        # --------------------------------------

        try:

            consistency = (
                check_model_consistency(
                    main_probability,
                    reconstructed_probability,
                )
            )

            status = consistency[
                "status"
            ]

            main_class = consistency[
                "main_class"
            ]

            gradcam_class = consistency[
                "gradcam_class"
            ]

            explanation_available = (
                "YES"
            )

            failure_reason = ""


        except RuntimeError as error:

            status = "FAILED"

            explanation_available = (
                "NO"
            )

            failure_reason = str(
                error
            )

            main_class = (
                "Tuberculosis"
                if main_probability >= 0.5
                else "Normal"
            )

            gradcam_class = (
                "Tuberculosis"
                if reconstructed_probability >= 0.5
                else "Normal"
            )


        rows.append({
            "image_id":
                anonymous_id,

            "main_probability":
                main_probability,

            "gradcam_probability":
                reconstructed_probability,

            "absolute_difference":
                difference,

            "main_class":
                main_class,

            "gradcam_class":
                gradcam_class,

            "consistency_status":
                status,

            "explanation_available":
                explanation_available,

            "failure_reason":
                failure_reason,
        })


    except Exception as error:

        rows.append({
            "image_id":
                anonymous_id,

            "main_probability":
                "",

            "gradcam_probability":
                "",

            "absolute_difference":
                "",

            "main_class":
                "",

            "gradcam_class":
                "",

            "consistency_status":
                "ERROR",

            "explanation_available":
                "NO",

            "failure_reason":
                str(error),
        })


# ============================================================
# SAVE RAW CSV
# ============================================================

with open(
    CSV_FILE,
    "w",
    newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "image_id",
            "main_probability",
            "gradcam_probability",
            "absolute_difference",
            "main_class",
            "gradcam_class",
            "consistency_status",
            "explanation_available",
            "failure_reason",
        ]
    )

    writer.writeheader()

    writer.writerows(
        rows
    )


# ============================================================
# SUMMARY STATISTICS
# ============================================================

valid_rows = [
    row
    for row in rows
    if row[
        "absolute_difference"
    ] != ""
]

differences = [
    float(
        row[
            "absolute_difference"
        ]
    )
    for row in valid_rows
]

pass_count = sum(
    row["consistency_status"] == "PASS"
    for row in rows
)

warning_count = sum(
    row["consistency_status"]
    == "PASS WITH WARNING"
    for row in rows
)

failed_count = sum(
    row["consistency_status"] == "FAILED"
    for row in rows
)

error_count = sum(
    row["consistency_status"] == "ERROR"
    for row in rows
)

available_count = sum(
    row["explanation_available"] == "YES"
    for row in rows
)

class_disagreements = sum(

    row["main_class"]
    and
    row["gradcam_class"]
    and
    row["main_class"]
    !=
    row["gradcam_class"]

    for row in rows
)


if differences:

    mean_difference = (
        statistics.mean(
            differences
        )
    )

    median_difference = (
        statistics.median(
            differences
        )
    )

    maximum_difference = max(
        differences
    )

else:

    mean_difference = 0.0
    median_difference = 0.0
    maximum_difference = 0.0


availability_percent = (
    available_count
    /
    len(rows)
    *
    100.0
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = f"""
======================================
TB GUARD - GRAD-CAM CONSISTENCY SUMMARY
======================================

Test completed:
{datetime.now().isoformat()}

Images tested:
{len(rows)}

Warning threshold:
{CONSISTENCY_WARNING_THRESHOLD:.4f}

Failure threshold:
{CONSISTENCY_FAILURE_THRESHOLD:.4f}

--------------------------------------
CONSISTENCY RESULTS
--------------------------------------

PASS:
{pass_count}

PASS WITH WARNING:
{warning_count}

FAILED:
{failed_count}

PROCESSING ERRORS:
{error_count}

Class disagreements:
{class_disagreements}

--------------------------------------
GRAD-CAM AVAILABILITY
--------------------------------------

Explanation available:
{available_count} / {len(rows)}

Availability rate:
{availability_percent:.2f}%

--------------------------------------
PROBABILITY DIFFERENCE
--------------------------------------

Mean absolute difference:
{mean_difference:.6f}

Median absolute difference:
{median_difference:.6f}

Maximum absolute difference:
{maximum_difference:.6f}

--------------------------------------

A Grad-CAM explanation is considered available only
when the authoritative prediction model and the
reconstructed Grad-CAM classification path satisfy
the deployed consistency safety checks.
"""


with open(
    SUMMARY_FILE,
    "w"
) as file:

    file.write(
        summary
    )


print()
print(summary)

print(
    "Raw results saved to:"
)

print(
    CSV_FILE
)

print()

print(
    "Summary saved to:"
)

print(
    SUMMARY_FILE
)
