import csv
import io
import json
import traceback
import uuid
from pathlib import Path

from flask import (
    Flask,
    abort,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)

from werkzeug.utils import secure_filename

from screening_engine import analyse_xray

from database import (
    init_database,
    create_screening,
    complete_screening,
    fail_screening,
    get_screening,
    get_all_screenings,
    delete_screening,
)

from storage import (
    SCREENINGS_DIR,
    ORIGINALS_DIR,
    GRADCAM_DIR,
    initialise_storage,
    save_uploaded_image,
    validate_gradcam_file,
    delete_managed_file,
    managed_file_exists,
    require_original_path,
    require_gradcam_path,
    validate_extension,
    audit_storage,
    clean_temporary_files,
)


# ============================================================
# APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# CONFIGURATION
# ============================================================

MAX_UPLOAD_SIZE = 15 * 1024 * 1024

app.config[
    "MAX_CONTENT_LENGTH"
] = MAX_UPLOAD_SIZE


MODEL_NAME = "MobileNetV3"

MODEL_VERSION = "1.0"


# ============================================================
# INITIALISE STORAGE
# ============================================================

initialise_storage()


removed_temp_files = (
    clean_temporary_files()
)


if removed_temp_files:

    print(
        "Removed stale temporary upload files:",
        removed_temp_files,
    )


# ============================================================
# INITIALISE DATABASE
# ============================================================

init_database()


# ============================================================
# HELPERS
# ============================================================

def allowed_file(
    filename
):

    try:

        validate_extension(
            filename
        )

        return True


    except ValueError:

        return False


def make_screening_id():

    return (
        "SCR-"
        +
        uuid.uuid4()
        .hex[:12]
        .upper()
    )


def safe_probability(
    value
):

    if value is None:

        return None


    try:

        if isinstance(
            value,
            str,
        ):

            cleaned = (
                value
                .strip()
            )


            percentage = (
                cleaned
                .endswith("%")
            )


            cleaned = (
                cleaned
                .replace(
                    "%",
                    "",
                )
            )


            value = float(
                cleaned
            )


            if percentage:

                value = (
                    value
                    /
                    100.0
                )


        else:

            value = float(
                value
            )


    except (
        TypeError,
        ValueError,
    ):

        return None


    if value > 1.0:

        value = (
            value
            /
            100.0
        )


    value = max(
        0.0,
        min(
            1.0,
            value,
        ),
    )


    return value


def normalise_key(
    key
):

    return (
        str(
            key
        )
        .strip()
        .lower()
        .replace(
            " ",
            "_",
        )
        .replace(
            "-",
            "_",
        )
    )


def first_result_value(
    values,
    *keys,
):

    for key in keys:

        key = normalise_key(
            key
        )


        if (
            key in values
            and
            values[
                key
            ]
            is not None
        ):

            return values[
                key
            ]


    return None


def extract_screening_result(
    engine_result
):

    if not isinstance(
        engine_result,
        dict,
    ):

        raise TypeError(
            "analyse_xray() must return a dictionary."
        )


    values = {

        normalise_key(
            key
        ):
            value

        for key, value
        in engine_result.items()
    }


    prediction = (
        first_result_value(

            values,

            "prediction",

            "predicted_class",

            "predicted_label",

            "class",

            "label",

            "result",
        )
    )


    tb_probability = (
        safe_probability(

            first_result_value(

                values,

                "tb_probability",

                "tb_prob",

                "probability",

                "main_model_probability",

                "model_probability",

                "prediction_probability",

                "score",
            )
        )
    )


    normal_probability = (
        safe_probability(

            first_result_value(

                values,

                "normal_probability",

                "normal_prob",

                "negative_probability",
            )
        )
    )


    confidence = (
        safe_probability(

            first_result_value(

                values,

                "confidence",

                "prediction_confidence",

                "model_confidence",
            )
        )
    )


    inference_ms = (
        first_result_value(

            values,

            "inference_ms",

            "inference_time_ms",

            "inference_time",
        )
    )


    if inference_ms is not None:

        try:

            inference_ms = float(
                inference_ms
            )


        except (
            TypeError,
            ValueError,
        ):

            inference_ms = None


    prediction_lower = (
        str(
            prediction
            or
            ""
        )
        .strip()
        .lower()
    )


    # ========================================================
    # DERIVE TB PROBABILITY
    # ========================================================

    if (
        tb_probability is None
        and
        confidence is not None
    ):

        if (
            "tuberculosis"
            in prediction_lower
            or
            prediction_lower
            ==
            "tb"
            or
            "positive"
            in prediction_lower
        ):

            tb_probability = (
                confidence
            )


        elif (
            "normal"
            in prediction_lower
            or
            "negative"
            in prediction_lower
        ):

            tb_probability = (
                1.0
                -
                confidence
            )


    # ========================================================
    # NORMAL PROBABILITY
    # ========================================================

    if (
        tb_probability is not None
        and
        normal_probability is None
    ):

        normal_probability = (
            1.0
            -
            tb_probability
        )


    if (
        normal_probability is not None
        and
        tb_probability is None
    ):

        tb_probability = (
            1.0
            -
            normal_probability
        )


    # ========================================================
    # PREDICTION
    # ========================================================

    if (
        prediction is None
        and
        tb_probability is not None
    ):

        if tb_probability >= 0.5:

            prediction = (
                "Tuberculosis"
            )


        else:

            prediction = (
                "Normal"
            )


    # ========================================================
    # CONFIDENCE
    # ========================================================

    if (
        confidence is None
        and
        tb_probability is not None
        and
        normal_probability is not None
    ):

        confidence = max(
            tb_probability,
            normal_probability,
        )


    if prediction is None:

        raise ValueError(
            "The screening engine did not return a prediction."
        )


    if tb_probability is None:

        raise ValueError(
            "The screening engine did not return "
            "a TB probability."
        )


    if normal_probability is None:

        normal_probability = (
            1.0
            -
            tb_probability
        )


    if confidence is None:

        confidence = max(
            tb_probability,
            normal_probability,
        )


    return {

        "prediction":
            str(
                prediction
            ),

        "tb_probability":
            float(
                tb_probability
            ),

        "normal_probability":
            float(
                normal_probability
            ),

        "confidence":
            float(
                confidence
            ),

        "inference_ms":
            inference_ms,
    }


def get_result_category(
    prediction,
    status,
):

    prediction = (
        prediction
        or
        ""
    ).strip().lower()


    status = (
        status
        or
        ""
    ).strip().lower()


    if status == "failed":

        return "failed"


    normal_phrases = (

        "normal",

        "negative",

        "no tuberculosis",

        "no tb",

        "tb negative",

        "non-tb",
    )


    for phrase in normal_phrases:

        if phrase in prediction:

            return "normal"


    if (
        prediction == "tb"
        or
        "tuberculosis"
        in prediction
        or
        "tb positive"
        in prediction
        or
        "positive"
        in prediction
    ):

        return "tb"


    return "other"


# ============================================================
# HOME
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# SCREEN X-RAY
# ============================================================

@app.route(
    "/screen",
    methods=[
        "POST",
    ],
)
def screen():

    # ========================================================
    # VALIDATE UPLOAD
    # ========================================================

    if (
        "xray"
        not in request.files
    ):

        return render_template(

            "index.html",

            error=(
                "Please select a chest X-ray."
            ),
        )


    uploaded_file = (
        request.files[
            "xray"
        ]
    )


    if not uploaded_file.filename:

        return render_template(

            "index.html",

            error=(
                "Please select a chest X-ray."
            ),
        )


    original_filename = (
        secure_filename(
            uploaded_file.filename
        )
    )


    if not original_filename:

        return render_template(

            "index.html",

            error=(
                "The uploaded filename is invalid."
            ),
        )


    if not allowed_file(
        original_filename
    ):

        return render_template(

            "index.html",

            error=(
                "Unsupported image type. "
                "Please upload a PNG, JPG or JPEG image."
            ),
        )


    # ========================================================
    # CREATE SCREENING ID
    # ========================================================

    screening_id = (
        make_screening_id()
    )


    extension = (
        Path(
            original_filename
        )
        .suffix
        .lower()
    )


    stored_original_filename = (

        f"{screening_id}"
        f"{extension}"
    )


    gradcam_filename = (

        f"{screening_id}"
        "_gradcam.png"
    )


    original_path = (

        ORIGINALS_DIR
        /
        stored_original_filename
    )


    gradcam_path = (

        GRADCAM_DIR
        /
        gradcam_filename
    )


    # ========================================================
    # SAFE ATOMIC IMAGE STORAGE
    # ========================================================

    try:

        upload_metadata = (
            save_uploaded_image(

                uploaded_file,

                original_path,
            )
        )


        original_path = Path(
            upload_metadata[
                "path"
            ]
        )


        print()
        print(
            "========================================"
        )
        print(
            "UPLOAD VERIFIED"
        )
        print(
            "========================================"
        )

        print(
            "Format:",
            upload_metadata[
                "format"
            ],
        )

        print(
            "Dimensions:",
            (
                f"{upload_metadata['width']}"
                " x "
                f"{upload_metadata['height']}"
            ),
        )

        print(
            "Size:",
            upload_metadata[
                "size_bytes"
            ],
            "bytes",
        )

        print(
            "SHA-256:",
            upload_metadata[
                "sha256"
            ],
        )

        print(
            "========================================"
        )
        print()


    except Exception as exc:

        traceback.print_exc()


        return render_template(

            "index.html",

            error=(
                "The uploaded X-ray was rejected. "
                +
                str(
                    exc
                )
            ),
        )


    # ========================================================
    # CREATE DATABASE RECORD
    # ========================================================

    try:

        create_screening(

            screening_id=
                screening_id,

            original_filename=
                original_filename,

            stored_filename=
                stored_original_filename,

            image_path=
                str(
                    original_path
                ),

            model_name=
                MODEL_NAME,

            model_version=
                MODEL_VERSION,
        )


    except Exception as exc:

        traceback.print_exc()


        try:

            delete_managed_file(
                original_path
            )

        except Exception:

            traceback.print_exc()


        return render_template(

            "index.html",

            error=(
                "The screening record "
                "could not be created. "
                +
                str(
                    exc
                )
            ),
        )


    # ========================================================
    # SCREENING
    # ========================================================

    try:

        print()
        print(
            "========================================"
        )
        print(
            "STARTING SCREENING"
        )
        print(
            "========================================"
        )

        print(
            "Screening ID:",
            screening_id,
        )

        print(
            "Image:",
            original_path,
        )

        print(
            "Grad-CAM:",
            gradcam_path,
        )

        print(
            "========================================"
        )
        print()


        # ====================================================
        # RUN SCREENING ENGINE
        # ====================================================

        engine_result = (
            analyse_xray(

                str(
                    original_path
                ),

                str(
                    gradcam_path
                ),
            )
        )


        # ====================================================
        # NORMALISE MAIN RESULT
        # ====================================================

        parsed_result = (
            extract_screening_result(
                engine_result
            )
        )


        prediction = (
            parsed_result[
                "prediction"
            ]
        )


        tb_probability = (
            parsed_result[
                "tb_probability"
            ]
        )


        normal_probability = (
            parsed_result[
                "normal_probability"
            ]
        )


        confidence = (
            parsed_result[
                "confidence"
            ]
        )


        inference_ms = (
            parsed_result[
                "inference_ms"
            ]
        )


        # ====================================================
        # GRAD-CAM AVAILABILITY
        # ====================================================

        gradcam_available = bool(

            engine_result.get(
                "gradcam_available",
                False,
            )
        )


        gradcam_reason = (
            engine_result.get(
                "gradcam_reason"
            )
        )


        consistency_status = (
            engine_result.get(
                "consistency_status"
            )
        )


        consistency_difference = (
            engine_result.get(
                "consistency_difference"
            )
        )


        reconstructed_probability = (
            engine_result.get(
                "reconstructed_probability"
            )
        )


        saved_gradcam_path = None


        # ====================================================
        # VALIDATE GENERATED GRAD-CAM
        #
        # Even if screening_engine says it succeeded, the file
        # is independently checked before its path is stored.
        # ====================================================

        if gradcam_available:

            try:

                gradcam_metadata = (
                    validate_gradcam_file(
                        gradcam_path
                    )
                )


                saved_gradcam_path = (
                    gradcam_metadata[
                        "path"
                    ]
                )


                print(
                    "Grad-CAM SHA-256:",
                    gradcam_metadata[
                        "sha256"
                    ],
                )


            except Exception as exc:

                gradcam_available = False

                saved_gradcam_path = None

                gradcam_reason = (
                    "Generated Grad-CAM failed "
                    "storage validation. "
                    +
                    str(
                        exc
                    )
                )


                try:

                    if managed_file_exists(
                        gradcam_path
                    ):

                        delete_managed_file(
                            gradcam_path
                        )


                except Exception:

                    traceback.print_exc()


        else:

            try:

                if managed_file_exists(
                    gradcam_path
                ):

                    delete_managed_file(
                        gradcam_path
                    )


            except Exception:

                traceback.print_exc()


        # ====================================================
        # ENSURE RESULT JSON MATCHES FINAL STORAGE STATE
        # ====================================================

        engine_result[
            "gradcam_available"
        ] = gradcam_available


        engine_result[
            "gradcam_reason"
        ] = gradcam_reason


        if saved_gradcam_path is not None:

            engine_result[
                "gradcam_path"
            ] = saved_gradcam_path


        else:

            engine_result[
                "gradcam_path"
            ] = None


        engine_result[
            "original_sha256"
        ] = upload_metadata[
            "sha256"
        ]


        engine_result[
            "original_size_bytes"
        ] = upload_metadata[
            "size_bytes"
        ]


        engine_result[
            "original_width"
        ] = upload_metadata[
            "width"
        ]


        engine_result[
            "original_height"
        ] = upload_metadata[
            "height"
        ]


        result_json = json.dumps(

            engine_result,

            default=str,
        )


        # ====================================================
        # SAVE COMPLETED SCREENING
        # ====================================================

        complete_screening(

            screening_id=
                screening_id,

            prediction=
                prediction,

            confidence=
                confidence,

            inference_ms=
                inference_ms,

            tb_probability=
                tb_probability,

            normal_probability=
                normal_probability,

            gradcam_path=
                saved_gradcam_path,

            result_json=
                result_json,
        )


        print()
        print(
            "========================================"
        )
        print(
            "SCREENING SAVED SUCCESSFULLY"
        )
        print(
            "========================================"
        )

        print(
            "Screening ID:",
            screening_id,
        )

        print(
            "Status: completed"
        )

        print(
            "Prediction:",
            prediction,
        )

        print(
            "Grad-CAM available:",
            gradcam_available,
        )

        print(
            "Consistency:",
            consistency_status,
        )


        if consistency_difference is not None:

            print(
                "Consistency difference:",
                consistency_difference,
            )


        if reconstructed_probability is not None:

            print(
                "Reconstructed probability:",
                reconstructed_probability,
            )


        if not gradcam_available:

            print(
                "Grad-CAM reason:",
                gradcam_reason,
            )


        print(
            "========================================"
        )
        print()


    # ========================================================
    # ACTUAL SCREENING FAILURE
    # ========================================================

    except Exception as exc:

        traceback.print_exc()


        error_message = str(
            exc
        )


        try:

            fail_screening(

                screening_id,

                error_message,
            )


        except Exception:

            traceback.print_exc()


        try:

            if managed_file_exists(
                gradcam_path
            ):

                delete_managed_file(
                    gradcam_path
                )


        except Exception:

            traceback.print_exc()


        return redirect(

            url_for(

                "screening_result",

                screening_id=
                    screening_id,
            )
        )


    return redirect(

        url_for(

            "screening_result",

            screening_id=
                screening_id,
        )
    )


# ============================================================
# RESULT PAGE
# ============================================================

@app.route(
    "/result/<screening_id>"
)
def screening_result(
    screening_id
):

    screening = (
        get_screening(
            screening_id
        )
    )


    if screening is None:

        abort(
            404
        )


    screening = dict(
        screening
    )


    original_url = None

    gradcam_url = None


    # ========================================================
    # ORIGINAL IMAGE
    # ========================================================

    image_path = (
        screening.get(
            "image_path"
        )
    )


    if (
        image_path
        and
        managed_file_exists(
            image_path
        )
    ):

        try:

            safe_path = (
                require_original_path(
                    image_path
                )
            )


            original_url = (

                url_for(

                    "view_original",

                    filename=
                        safe_path.name,
                )
            )


        except ValueError:

            original_url = None


    # ========================================================
    # RESULT JSON
    # ========================================================

    result_json_data = {}


    raw_result_json = (
        screening.get(
            "result_json"
        )
    )


    if raw_result_json:

        try:

            result_json_data = (
                json.loads(
                    raw_result_json
                )
            )


        except (
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ):

            result_json_data = {}


    # ========================================================
    # GRAD-CAM
    # ========================================================

    saved_gradcam_path = (
        screening.get(
            "gradcam_path"
        )
    )


    if (
        saved_gradcam_path
        and
        managed_file_exists(
            saved_gradcam_path
        )
    ):

        try:

            safe_gradcam = (
                require_gradcam_path(
                    saved_gradcam_path
                )
            )


            gradcam_url = (

                url_for(

                    "view_gradcam",

                    filename=
                        safe_gradcam.name,
                )
            )


        except ValueError:

            gradcam_url = None


    result = {

        "screening_id":
            screening.get(
                "screening_id"
            ),

        "prediction":
            screening.get(
                "prediction"
            ),

        "confidence":
            screening.get(
                "confidence"
            ),

        "tb_probability":
            screening.get(
                "tb_probability"
            ),

        "normal_probability":
            screening.get(
                "normal_probability"
            ),

        "status":
            screening.get(
                "status"
            ),

        "error_message":
            screening.get(
                "error_message"
            ),

        "model_name":
            screening.get(
                "model_name"
            ),

        "model_version":
            screening.get(
                "model_version"
            ),

        "created_at":
            screening.get(
                "created_at"
            ),

        "gradcam_available":
            result_json_data.get(
                "gradcam_available"
            ),

        "gradcam_reason":
            result_json_data.get(
                "gradcam_reason"
            ),

        "consistency_status":
            result_json_data.get(
                "consistency_status"
            ),

        "consistency_difference":
            result_json_data.get(
                "consistency_difference"
            ),

        "reconstructed_probability":
            result_json_data.get(
                "reconstructed_probability"
            ),
    }


    return render_template(

        "result.html",

        result=
            result,

        screening=
            screening,

        screening_id=
            screening_id,

        original_url=
            original_url,

        gradcam_url=
            gradcam_url,
    )


# ============================================================
# VIEW ORIGINAL
# ============================================================

@app.route(
    "/screenings/originals/<filename>"
)
def view_original(
    filename
):

    filename = secure_filename(
        filename
    )


    if not filename:

        abort(
            404
        )


    try:

        path = require_original_path(

            ORIGINALS_DIR
            /
            filename
        )


    except ValueError:

        abort(
            404
        )


    if not path.is_file():

        abort(
            404
        )


    return send_file(

        str(
            path
        ),

        as_attachment=
            False,
    )


# ============================================================
# VIEW GRAD-CAM
# ============================================================

@app.route(
    "/screenings/gradcam/<filename>"
)
def view_gradcam(
    filename
):

    filename = secure_filename(
        filename
    )


    if not filename:

        abort(
            404
        )


    try:

        path = require_gradcam_path(

            GRADCAM_DIR
            /
            filename
        )


    except ValueError:

        abort(
            404
        )


    if not path.is_file():

        abort(
            404
        )


    return send_file(

        str(
            path
        ),

        as_attachment=
            False,
    )


# ============================================================
# SCREENING DATABASE
# ============================================================

@app.route(
    "/database"
)
def screening_database():

    rows = (
        get_all_screenings()
    )


    screenings = []


    stats = {

        "total":
            0,

        "tb":
            0,

        "normal":
            0,

        "failed":
            0,
    }


    for row in rows:

        screening = dict(
            row
        )


        prediction = (
            screening.get(
                "prediction"
            )
            or
            ""
        )


        status = (
            screening.get(
                "status"
            )
            or
            ""
        )


        category = (
            get_result_category(
                prediction,
                status,
            )
        )


        screening[
            "result_category"
        ] = category


        result_data = {}


        raw_result_json = (
            screening.get(
                "result_json"
            )
        )


        if raw_result_json:

            try:

                result_data = (
                    json.loads(
                        raw_result_json
                    )
                )


            except (
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ):

                result_data = {}


        screening[
            "gradcam_available"
        ] = result_data.get(
            "gradcam_available"
        )


        screening[
            "gradcam_reason"
        ] = result_data.get(
            "gradcam_reason"
        )


        screening[
            "consistency_status"
        ] = result_data.get(
            "consistency_status"
        )


        gradcam_filename = None


        gradcam_path = (
            screening.get(
                "gradcam_path"
            )
        )


        if (
            gradcam_path
            and
            managed_file_exists(
                gradcam_path
            )
        ):

            try:

                gradcam_filename = (
                    require_gradcam_path(
                        gradcam_path
                    )
                    .name
                )


            except ValueError:

                gradcam_filename = None


        screening[
            "gradcam_filename"
        ] = gradcam_filename


        screenings.append(
            screening
        )


        stats[
            "total"
        ] += 1


        if category == "tb":

            stats[
                "tb"
            ] += 1


        elif category == "normal":

            stats[
                "normal"
            ] += 1


        if (
            status
            .strip()
            .lower()
            ==
            "failed"
        ):

            stats[
                "failed"
            ] += 1


    return render_template(

        "screening_database.html",

        screenings=
            screenings,

        stats=
            stats,
    )


# ============================================================
# DATABASE SHORTCUT
# ============================================================

@app.route(
    "/screenings"
)
def screenings_redirect():

    return redirect(

        url_for(
            "screening_database"
        )
    )


# ============================================================
# DOWNLOAD ORIGINAL
# ============================================================

@app.route(
    "/database/<screening_id>/download/original"
)
def download_original(
    screening_id
):

    screening = (
        get_screening(
            screening_id
        )
    )


    if screening is None:

        abort(
            404
        )


    screening = dict(
        screening
    )


    image_path = (
        screening.get(
            "image_path"
        )
    )


    if not image_path:

        abort(
            404
        )


    try:

        path = require_original_path(
            image_path
        )


    except ValueError:

        abort(
            404
        )


    if not path.is_file():

        abort(
            404
        )


    download_name = secure_filename(

        screening.get(
            "original_filename"
        )
        or
        path.name
    )


    return send_file(

        str(
            path
        ),

        as_attachment=
            True,

        download_name=
            download_name,
    )


# ============================================================
# DOWNLOAD GRAD-CAM
# ============================================================

@app.route(
    "/database/<screening_id>/download/gradcam"
)
def download_gradcam(
    screening_id
):

    screening = (
        get_screening(
            screening_id
        )
    )


    if screening is None:

        abort(
            404
        )


    screening = dict(
        screening
    )


    gradcam_path = (
        screening.get(
            "gradcam_path"
        )
    )


    if not gradcam_path:

        abort(
            404
        )


    try:

        path = require_gradcam_path(
            gradcam_path
        )


    except ValueError:

        abort(
            404
        )


    if not path.is_file():

        abort(
            404
        )


    return send_file(

        str(
            path
        ),

        as_attachment=
            True,

        download_name=
            (
                f"{screening_id}"
                "_gradcam.png"
            ),
    )


# ============================================================
# DOWNLOAD RECORD
# ============================================================

@app.route(
    "/database/<screening_id>/download/record"
)
def download_record(
    screening_id
):

    screening = (
        get_screening(
            screening_id
        )
    )


    if screening is None:

        abort(
            404
        )


    screening = dict(
        screening
    )


    output = io.StringIO()


    writer = csv.writer(
        output
    )


    writer.writerow(
        [
            "Field",
            "Value",
        ]
    )


    fields = [

        "screening_id",

        "original_filename",

        "prediction",

        "confidence",

        "tb_probability",

        "normal_probability",

        "model_name",

        "model_version",

        "status",

        "error_message",

        "inference_ms",

        "result_json",

        "created_at",

        "updated_at",
    ]


    for field in fields:

        writer.writerow(
            [
                field,
                screening.get(
                    field
                ),
            ]
        )


    csv_bytes = io.BytesIO(

        output
        .getvalue()
        .encode(
            "utf-8"
        )
    )


    csv_bytes.seek(
        0
    )


    return send_file(

        csv_bytes,

        mimetype=
            "text/csv",

        as_attachment=
            True,

        download_name=
            f"{screening_id}.csv",
    )


# ============================================================
# DELETE SCREENING
# ============================================================

@app.route(
    "/database/<screening_id>/delete",
    methods=[
        "POST",
    ],
)
def remove_screening(
    screening_id
):

    screening = (
        get_screening(
            screening_id
        )
    )


    if screening is None:

        abort(
            404
        )


    screening = dict(
        screening
    )


    original_path = (
        screening.get(
            "image_path"
        )
    )


    gradcam_path = (
        screening.get(
            "gradcam_path"
        )
    )


    # ========================================================
    # DATABASE DELETE FIRST
    #
    # File cleanup occurs only after the record deletion
    # succeeds.
    #
    # Any leftover file can later be detected by audit_storage.
    # ========================================================

    deleted = (
        delete_screening(
            screening_id
        )
    )


    if deleted:

        if original_path:

            try:

                delete_managed_file(
                    original_path
                )


            except Exception as exc:

                print(
                    "Original file cleanup warning:",
                    exc,
                )


        if gradcam_path:

            try:

                delete_managed_file(
                    gradcam_path
                )


            except Exception as exc:

                print(
                    "Grad-CAM cleanup warning:",
                    exc,
                )


    return redirect(

        url_for(
            "screening_database"
        )
    )


# ============================================================
# STORAGE AUDIT HELPER
# ============================================================

def run_storage_audit():

    rows = (
        get_all_screenings()
    )


    report = (
        audit_storage(
            rows
        )
    )


    return report


# ============================================================
# 404
# ============================================================

@app.errorhandler(
    404
)
def not_found(
    error
):

    return (
        """
        <!DOCTYPE html>

        <html>

        <head>

            <title>
                TB Guard | Not Found
            </title>

        </head>

        <body style="
            font-family: Arial, sans-serif;
            padding: 40px;
            background: #f7f5f0;
            color: #20302e;
        ">

            <h1>
                Record not found
            </h1>

            <p>
                The requested screening or image
                could not be found.
            </p>

            <p>
                <a href="/">
                    Return to TB Guard
                </a>
            </p>

        </body>

        </html>
        """,

        404,
    )


# ============================================================
# FILE TOO LARGE
# ============================================================

@app.errorhandler(
    413
)
def file_too_large(
    error
):

    return (

        render_template(

            "index.html",

            error=(
                "The uploaded image is too large. "
                "Maximum size is 15 MB."
            ),
        ),

        413,
    )


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    print()

    print(
        "=========================================="
    )

    print(
        "TB Raspberry Pi Screening Application"
    )

    print(
        "=========================================="
    )


    print(
        "Database:",
        BASE_DIR
        /
        "screening.db",
    )


    print(
        "Original images:",
        ORIGINALS_DIR,
    )


    print(
        "Grad-CAM images:",
        GRADCAM_DIR,
    )


    # ========================================================
    # STORAGE AUDIT
    # ========================================================

    try:

        audit_report = (
            run_storage_audit()
        )


        print()
        print(
            "Storage audit:"
        )


        print(
            "Missing files:",
            len(
                audit_report[
                    "missing_database_files"
                ]
            ),
        )


        print(
            "Orphan originals:",
            len(
                audit_report[
                    "orphan_originals"
                ]
            ),
        )


        print(
            "Orphan Grad-CAMs:",
            len(
                audit_report[
                    "orphan_gradcams"
                ]
            ),
        )


    except Exception as exc:

        print(
            "Storage audit warning:",
            exc,
        )


    print()

    print(
        "Home:"
    )


    print(
        "http://127.0.0.1:5000/"
    )


    print()

    print(
        "Screening database:"
    )


    print(
        "http://127.0.0.1:5000/database"
    )


    print()


    app.run(

        host=
            "0.0.0.0",

        port=
            5000,

        debug=
            False,
    )
