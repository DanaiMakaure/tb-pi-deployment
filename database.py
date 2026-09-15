import json
import math
import re
import sqlite3

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATABASE_PATH = (
    BASE_DIR
    /
    "screening.db"
)

BACKUP_DIR = (
    BASE_DIR
    /
    "backups"
)


# ============================================================
# SQLITE SETTINGS
# ============================================================

DATABASE_TIMEOUT_SECONDS = 30

BUSY_TIMEOUT_MS = 30000


VALID_STATUSES = {
    "processing",
    "completed",
    "failed",
}


SAFE_IDENTIFIER = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*$"
)


# ============================================================
# VALIDATION HELPERS
# ============================================================

def _require_nonempty_text(
    value,
    field_name,
    max_length=None
):

    if value is None:

        raise ValueError(
            f"{field_name} is required."
        )


    value = str(
        value
    ).strip()


    if not value:

        raise ValueError(
            f"{field_name} cannot be empty."
        )


    if (
        max_length is not None
        and
        len(
            value
        )
        >
        max_length
    ):

        raise ValueError(
            f"{field_name} is too long."
        )


    return value


def _optional_text(
    value,
    field_name,
    max_length=None
):

    if value is None:

        return None


    value = str(
        value
    ).strip()


    if not value:

        return None


    if (
        max_length is not None
        and
        len(
            value
        )
        >
        max_length
    ):

        raise ValueError(
            f"{field_name} is too long."
        )


    return value


def _validate_probability(
    value,
    field_name
):

    if value is None:

        raise ValueError(
            f"{field_name} is required."
        )


    try:

        value = float(
            value
        )


    except (
        TypeError,
        ValueError
    ):

        raise ValueError(
            f"{field_name} must be numeric."
        )


    if not math.isfinite(
        value
    ):

        raise ValueError(
            f"{field_name} must be finite."
        )


    if not (
        0.0
        <=
        value
        <=
        1.0
    ):

        raise ValueError(
            f"{field_name} must be between 0 and 1."
        )


    return value


def _validate_inference_ms(
    value
):

    if value is None:

        return None


    try:

        value = float(
            value
        )


    except (
        TypeError,
        ValueError
    ):

        raise ValueError(
            "inference_ms must be numeric."
        )


    if (
        not math.isfinite(
            value
        )
        or
        value < 0.0
    ):

        raise ValueError(
            "inference_ms must be a finite "
            "non-negative value."
        )


    return value


def _validate_result_json(
    value
):

    if value is None:

        return None


    if not isinstance(
        value,
        str
    ):

        raise ValueError(
            "result_json must be a JSON string."
        )


    try:

        json.loads(
            value
        )


    except json.JSONDecodeError as exc:

        raise ValueError(
            "result_json contains invalid JSON."
        ) from exc


    return value


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_db_connection():

    connection = sqlite3.connect(

        str(
            DATABASE_PATH
        ),

        timeout=
            DATABASE_TIMEOUT_SECONDS
    )


    connection.row_factory = (
        sqlite3.Row
    )


    # --------------------------------------------------------
    # FOREIGN KEY PROTECTION
    #
    # We do not currently have related tables, but this makes
    # foreign-key enforcement automatic when more tables are
    # added later.
    # --------------------------------------------------------

    connection.execute(
        "PRAGMA foreign_keys = ON"
    )


    # --------------------------------------------------------
    # BUSY TIMEOUT
    #
    # Prevent immediate "database is locked" failures when
    # two requests briefly overlap.
    # --------------------------------------------------------

    connection.execute(
        f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}"
    )


    # --------------------------------------------------------
    # DURABILITY
    #
    # FULL favours data safety over a small amount of write
    # performance.
    # --------------------------------------------------------

    connection.execute(
        "PRAGMA synchronous = FULL"
    )


    return connection


# ============================================================
# SAFE READ CONNECTION
# ============================================================

@contextmanager
def database_connection():

    connection = (
        get_db_connection()
    )


    try:

        yield connection


    finally:

        connection.close()


# ============================================================
# SAFE WRITE TRANSACTION
#
# Every database write now has:
#
# BEGIN
#     ↓
# change
#     ↓
# success → COMMIT
#
# error
#     ↓
# ROLLBACK
#
# connection always closes.
# ============================================================

@contextmanager
def write_transaction():

    connection = (
        get_db_connection()
    )


    try:

        # ----------------------------------------------------
        # BEGIN IMMEDIATE
        #
        # Reserve the SQLite writer before modifying anything.
        # This reduces partially-started write operations.
        # ----------------------------------------------------

        connection.execute(
            "BEGIN IMMEDIATE"
        )


        yield connection


        connection.commit()


    except Exception:

        connection.rollback()

        raise


    finally:

        connection.close()


# ============================================================
# SAFE SQL IDENTIFIER
# ============================================================

def _validate_identifier(
    identifier
):

    identifier = str(
        identifier
    )


    if not SAFE_IDENTIFIER.fullmatch(
        identifier
    ):

        raise ValueError(
            "Unsafe SQL identifier: "
            f"{identifier}"
        )


    return identifier


# ============================================================
# CHECK COLUMN
# ============================================================

def column_exists(
    connection,
    table_name,
    column_name
):

    table_name = (
        _validate_identifier(
            table_name
        )
    )


    column_name = (
        _validate_identifier(
            column_name
        )
    )


    columns = connection.execute(

        f"""
        PRAGMA table_info(
            {table_name}
        )
        """

    ).fetchall()


    return any(

        column[
            "name"
        ]
        ==
        column_name

        for column
        in columns
    )


# ============================================================
# DATABASE INTEGRITY CHECK
# ============================================================

def check_database_integrity():

    with database_connection() as connection:

        result = connection.execute(

            """
            PRAGMA quick_check
            """

        ).fetchone()


        if result is None:

            raise RuntimeError(
                "SQLite integrity check "
                "returned no result."
            )


        status = str(
            result[
                0
            ]
        )


        if (
            status.lower()
            !=
            "ok"
        ):

            raise RuntimeError(
                "SQLite integrity check failed: "
                f"{status}"
            )


    return True


# ============================================================
# INITIALISE DATABASE
# ============================================================

def init_database():

    # ========================================================
    # ENABLE WAL MODE
    #
    # WAL = Write-Ahead Logging
    #
    # Benefits:
    #
    # - safer recovery after interruption
    # - readers do not block normal writes
    # - better Flask concurrency
    #
    # WAL mode remains enabled after restart.
    # ========================================================

    with database_connection() as connection:

        result = connection.execute(

            """
            PRAGMA journal_mode = WAL
            """

        ).fetchone()


        if (
            result is None
            or
            str(
                result[
                    0
                ]
            ).lower()
            !=
            "wal"
        ):

            raise RuntimeError(
                "Could not enable SQLite WAL mode."
            )


        connection.execute(
            "PRAGMA wal_autocheckpoint = 1000"
        )


    # ========================================================
    # CREATE / UPGRADE DATABASE
    # ========================================================

    with write_transaction() as connection:

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS screenings (

                id INTEGER
                    PRIMARY KEY
                    AUTOINCREMENT,

                screening_id TEXT
                    UNIQUE
                    NOT NULL,

                original_filename TEXT,

                stored_filename TEXT,

                image_path TEXT,

                gradcam_path TEXT,

                prediction TEXT,


                confidence REAL

                    CHECK (

                        confidence IS NULL

                        OR

                        (
                            confidence >= 0.0
                            AND
                            confidence <= 1.0
                        )
                    ),


                inference_ms REAL

                    CHECK (

                        inference_ms IS NULL

                        OR

                        inference_ms >= 0.0
                    ),


                tb_probability REAL

                    CHECK (

                        tb_probability IS NULL

                        OR

                        (
                            tb_probability >= 0.0
                            AND
                            tb_probability <= 1.0
                        )
                    ),


                normal_probability REAL

                    CHECK (

                        normal_probability IS NULL

                        OR

                        (
                            normal_probability >= 0.0
                            AND
                            normal_probability <= 1.0
                        )
                    ),


                model_name TEXT,

                model_version TEXT,


                status TEXT

                    NOT NULL

                    DEFAULT 'processing'

                    CHECK (

                        status IN (

                            'processing',

                            'completed',

                            'failed'
                        )
                    ),


                error_message TEXT,

                result_json TEXT,


                created_at TIMESTAMP

                    NOT NULL

                    DEFAULT CURRENT_TIMESTAMP,


                updated_at TIMESTAMP
            )
            """
        )


        # ====================================================
        # UPGRADE EXISTING DATABASE SAFELY
        #
        # No records are deleted.
        # ====================================================

        required_columns = {

            "original_filename":
                "TEXT",

            "stored_filename":
                "TEXT",

            "image_path":
                "TEXT",

            "gradcam_path":
                "TEXT",

            "prediction":
                "TEXT",

            "confidence":
                "REAL",

            "inference_ms":
                "REAL",

            "tb_probability":
                "REAL",

            "normal_probability":
                "REAL",

            "model_name":
                "TEXT",

            "model_version":
                "TEXT",

            "status":
                "TEXT",

            "error_message":
                "TEXT",

            "result_json":
                "TEXT",

            "updated_at":
                "TIMESTAMP",
        }


        for (
            column_name,
            column_type

        ) in required_columns.items():


            if not column_exists(

                connection,

                "screenings",

                column_name
            ):

                connection.execute(

                    f"""
                    ALTER TABLE screenings
                    ADD COLUMN
                    {column_name}
                    {column_type}
                    """
                )


        # ====================================================
        # INDEXES
        # ====================================================

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_screenings_screening_id

            ON screenings(
                screening_id
            )
            """
        )


        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_screenings_status

            ON screenings(
                status
            )
            """
        )


        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_screenings_created_at

            ON screenings(
                created_at
            )
            """
        )


    # ========================================================
    # CHECK DATABASE AFTER STARTUP
    # ========================================================

    check_database_integrity()


# ============================================================
# CREATE SCREENING
# ============================================================

def create_screening(
    screening_id,
    original_filename,
    stored_filename,
    image_path,
    model_name,
    model_version
):

    # ========================================================
    # VALIDATE
    # ========================================================

    screening_id = (
        _require_nonempty_text(

            screening_id,

            "screening_id",

            100
        )
    )


    original_filename = (
        _require_nonempty_text(

            original_filename,

            "original_filename",

            255
        )
    )


    stored_filename = (
        _require_nonempty_text(

            stored_filename,

            "stored_filename",

            255
        )
    )


    image_path = (
        _require_nonempty_text(

            image_path,

            "image_path",

            4096
        )
    )


    model_name = (
        _require_nonempty_text(

            model_name,

            "model_name",

            255
        )
    )


    model_version = (
        _require_nonempty_text(

            model_version,

            "model_version",

            255
        )
    )


    # ========================================================
    # INSERT
    # ========================================================

    with write_transaction() as connection:

        connection.execute(
            """
            INSERT INTO screenings (

                screening_id,

                original_filename,

                stored_filename,

                image_path,

                model_name,

                model_version,

                status,

                updated_at
            )

            VALUES (

                ?,

                ?,

                ?,

                ?,

                ?,

                ?,

                'processing',

                CURRENT_TIMESTAMP
            )
            """,

            (

                screening_id,

                original_filename,

                stored_filename,

                image_path,

                model_name,

                model_version,
            )
        )


# ============================================================
# COMPLETE SCREENING
# ============================================================

def complete_screening(
    screening_id,
    prediction,
    confidence,
    inference_ms,
    tb_probability,
    normal_probability,
    gradcam_path,
    result_json
):

    # ========================================================
    # VALIDATE
    # ========================================================

    screening_id = (
        _require_nonempty_text(

            screening_id,

            "screening_id",

            100
        )
    )


    prediction = (
        _require_nonempty_text(

            prediction,

            "prediction",

            100
        )
    )


    confidence = (
        _validate_probability(

            confidence,

            "confidence"
        )
    )


    tb_probability = (
        _validate_probability(

            tb_probability,

            "tb_probability"
        )
    )


    normal_probability = (
        _validate_probability(

            normal_probability,

            "normal_probability"
        )
    )


    inference_ms = (
        _validate_inference_ms(
            inference_ms
        )
    )


    gradcam_path = (
        _optional_text(

            gradcam_path,

            "gradcam_path",

            4096
        )
    )


    result_json = (
        _validate_result_json(
            result_json
        )
    )


    # ========================================================
    # PROBABILITY INTEGRITY
    # ========================================================

    probability_total = (

        tb_probability
        +
        normal_probability
    )


    if abs(

        probability_total
        -
        1.0

    ) > 0.0001:

        raise ValueError(

            "tb_probability and "
            "normal_probability must add "
            "up to approximately 1.0."
        )


    expected_confidence = max(

        tb_probability,

        normal_probability
    )


    if abs(

        confidence
        -
        expected_confidence

    ) > 0.0001:

        raise ValueError(

            "confidence does not match "
            "the stored class probabilities."
        )


    # ========================================================
    # UPDATE
    #
    # Only a PROCESSING screening can become COMPLETED.
    #
    # This prevents an already-finalised record from being
    # silently overwritten.
    # ========================================================

    with write_transaction() as connection:

        cursor = connection.execute(
            """
            UPDATE screenings

            SET

                prediction = ?,

                confidence = ?,

                inference_ms = ?,

                tb_probability = ?,

                normal_probability = ?,

                gradcam_path = ?,

                result_json = ?,

                status = 'completed',

                error_message = NULL,

                updated_at =
                    CURRENT_TIMESTAMP

            WHERE screening_id = ?

              AND status = 'processing'
            """,

            (

                prediction,

                confidence,

                inference_ms,

                tb_probability,

                normal_probability,

                gradcam_path,

                result_json,

                screening_id,
            )
        )


        # ====================================================
        # VERIFY EXACTLY ONE RECORD WAS UPDATED
        # ====================================================

        if cursor.rowcount != 1:

            existing = connection.execute(
                """
                SELECT
                    status

                FROM screenings

                WHERE screening_id = ?
                """,

                (
                    screening_id,
                )
            ).fetchone()


            if existing is None:

                raise LookupError(

                    "Cannot complete screening "
                    "because the record does not exist: "
                    f"{screening_id}"
                )


            raise RuntimeError(

                "Screening cannot be completed "
                "from status "
                f"'{existing['status']}'."
            )


# ============================================================
# FAIL SCREENING
# ============================================================

def fail_screening(
    screening_id,
    error_message
):

    screening_id = (
        _require_nonempty_text(

            screening_id,

            "screening_id",

            100
        )
    )


    error_message = (
        _require_nonempty_text(

            error_message,

            "error_message",

            4000
        )
    )


    with write_transaction() as connection:

        cursor = connection.execute(
            """
            UPDATE screenings

            SET

                status =
                    'failed',

                error_message =
                    ?,

                gradcam_path =
                    NULL,

                updated_at =
                    CURRENT_TIMESTAMP

            WHERE screening_id = ?

              AND status =
                    'processing'
            """,

            (
                error_message,

                screening_id,
            )
        )


        if cursor.rowcount != 1:

            existing = connection.execute(
                """
                SELECT
                    status

                FROM screenings

                WHERE screening_id = ?
                """,

                (
                    screening_id,
                )
            ).fetchone()


            if existing is None:

                raise LookupError(

                    "Cannot fail screening because "
                    "the record does not exist: "
                    f"{screening_id}"
                )


            # ------------------------------------------------
            # Already failed.
            #
            # Do not keep rewriting the same record.
            # ------------------------------------------------

            if (
                existing[
                    "status"
                ]
                ==
                "failed"
            ):

                return False


            raise RuntimeError(

                "Screening cannot be marked failed "
                "from status "
                f"'{existing['status']}'."
            )


        return True


# ============================================================
# GET ONE SCREENING
# ============================================================

def get_screening(
    screening_id
):

    screening_id = (
        _require_nonempty_text(

            screening_id,

            "screening_id",

            100
        )
    )


    with database_connection() as connection:

        screening = connection.execute(
            """
            SELECT

                id,

                screening_id,


                original_filename,

                stored_filename,


                image_path,

                gradcam_path,


                prediction,

                confidence,

                inference_ms,


                tb_probability,

                normal_probability,


                model_name,

                model_version,


                status,

                error_message,


                result_json,


                created_at,

                updated_at

            FROM screenings

            WHERE screening_id = ?
            """,

            (
                screening_id,
            )
        ).fetchone()


        return screening


# ============================================================
# GET ALL SCREENINGS
# ============================================================

def get_all_screenings():

    with database_connection() as connection:

        screenings = connection.execute(
            """
            SELECT

                id,

                screening_id,


                original_filename,

                stored_filename,


                image_path,

                gradcam_path,


                prediction,

                confidence,

                inference_ms,


                tb_probability,

                normal_probability,


                model_name,

                model_version,


                status,

                error_message,


                result_json,


                created_at,

                updated_at

            FROM screenings

            ORDER BY

                created_at DESC,

                id DESC
            """
        ).fetchall()


        return screenings


# ============================================================
# DELETE SCREENING
# ============================================================

def delete_screening(
    screening_id
):

    screening_id = (
        _require_nonempty_text(

            screening_id,

            "screening_id",

            100
        )
    )


    with write_transaction() as connection:

        cursor = connection.execute(
            """
            DELETE FROM screenings

            WHERE screening_id = ?
            """,

            (
                screening_id,
            )
        )


        return (
            cursor.rowcount
            ==
            1
        )


# ============================================================
# COUNT SCREENINGS
# ============================================================

def count_screenings():

    with database_connection() as connection:

        result = connection.execute(
            """
            SELECT

                COUNT(*) AS total

            FROM screenings
            """
        ).fetchone()


        return int(
            result[
                "total"
            ]
        )


# ============================================================
# SAFE DATABASE BACKUP
#
# IMPORTANT:
#
# Do NOT simply copy screening.db while Flask is running.
#
# Because WAL mode may contain committed data in:
#
# screening.db-wal
#
# SQLite's backup API creates a consistent snapshot.
# ============================================================

def backup_database(
    destination_path=None
):

    BACKUP_DIR.mkdir(

        parents=True,

        exist_ok=True
    )


    # ========================================================
    # AUTOMATIC BACKUP NAME
    # ========================================================

    if destination_path is None:

        timestamp = (

            datetime.now(
                timezone.utc
            )

            .strftime(
                "%Y%m%d_%H%M%S"
            )
        )


        destination_path = (

            BACKUP_DIR

            /

            (
                "screening_backup_"
                +
                timestamp
                +
                ".db"
            )
        )


    else:

        destination_path = Path(
            destination_path
        )


        destination_path.parent.mkdir(

            parents=True,

            exist_ok=True
        )


    destination_path = (
        destination_path.resolve()
    )


    # ========================================================
    # NEVER OVERWRITE LIVE DATABASE
    # ========================================================

    if (
        destination_path
        ==
        DATABASE_PATH.resolve()
    ):

        raise ValueError(

            "Backup destination cannot be "
            "the live screening database."
        )


    # ========================================================
    # SQLITE BACKUP API
    # ========================================================

    source = (
        get_db_connection()
    )


    backup = sqlite3.connect(

        str(
            destination_path
        )
    )


    try:

        source.backup(
            backup
        )


        backup.commit()


    finally:

        backup.close()

        source.close()


    # ========================================================
    # VERIFY BACKUP
    # ========================================================

    verify = sqlite3.connect(

        str(
            destination_path
        )
    )


    try:

        integrity = verify.execute(
            """
            PRAGMA quick_check
            """
        ).fetchone()


    finally:

        verify.close()


    if (
        integrity is None

        or

        str(
            integrity[
                0
            ]
        ).lower()
        !=
        "ok"
    ):

        raise RuntimeError(
            "Backup database failed "
            "its integrity check."
        )


    return destination_path


# ============================================================
# COMMAND-LINE TEST
# ============================================================

if __name__ == "__main__":

    init_database()


    print(
        "Database initialised successfully."
    )


    print(
        "Database:",
        DATABASE_PATH
    )


    print(
        "WAL mode: enabled"
    )


    print(
        "Integrity check: OK"
    )


    print(
        "Screening records:",
        count_screenings()
    )
