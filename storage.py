import hashlib
import os
import shutil
import warnings

from pathlib import Path

from PIL import (
    Image,
    UnidentifiedImageError,
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(
    __file__
).resolve().parent


SCREENINGS_DIR = (
    BASE_DIR
    /
    "screenings"
)


ORIGINALS_DIR = (
    SCREENINGS_DIR
    /
    "originals"
)


GRADCAM_DIR = (
    SCREENINGS_DIR
    /
    "gradcam"
)


TEMP_DIR = (
    SCREENINGS_DIR
    /
    "tmp"
)


# ============================================================
# FILE LIMITS
# ============================================================

MAX_UPLOAD_BYTES = (
    15
    *
    1024
    *
    1024
)


MAX_IMAGE_WIDTH = 12000

MAX_IMAGE_HEIGHT = 12000


ALLOWED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
}


ALLOWED_FORMATS = {
    "PNG",
    "JPEG",
}


# ============================================================
# CREATE SAFE STORAGE DIRECTORIES
# ============================================================

def initialise_storage():

    directories = [

        SCREENINGS_DIR,

        ORIGINALS_DIR,

        GRADCAM_DIR,

        TEMP_DIR,
    ]


    for directory in directories:

        directory.mkdir(

            parents=True,

            exist_ok=True,
        )


        # ----------------------------------------------------
        # Raspberry Pi / Linux:
        #
        # Owner can read/write/access.
        # Other system users cannot browse patient images.
        # ----------------------------------------------------

        try:

            os.chmod(
                directory,
                0o700,
            )

        except OSError:

            pass


# ============================================================
# PATH CONTAINMENT
#
# Prevent paths such as:
#
# ../../etc/passwd
#
# or symlinks escaping the screening directories.
# ============================================================

def _is_within(
    path,
    parent,
):

    path = Path(
        path
    ).resolve()


    parent = Path(
        parent
    ).resolve()


    try:

        path.relative_to(
            parent
        )

        return True


    except ValueError:

        return False


def require_original_path(
    path
):

    path = Path(
        path
    )


    if not _is_within(
        path,
        ORIGINALS_DIR,
    ):

        raise ValueError(
            "Unsafe original-image path."
        )


    return path.resolve()


def require_gradcam_path(
    path
):

    path = Path(
        path
    )


    if not _is_within(
        path,
        GRADCAM_DIR,
    ):

        raise ValueError(
            "Unsafe Grad-CAM path."
        )


    return path.resolve()


def require_managed_path(
    path
):

    path = Path(
        path
    )


    if _is_within(
        path,
        ORIGINALS_DIR,
    ):

        return path.resolve()


    if _is_within(
        path,
        GRADCAM_DIR,
    ):

        return path.resolve()


    raise ValueError(
        "Path is outside TB Guard managed storage."
    )


# ============================================================
# EXTENSION VALIDATION
# ============================================================

def validate_extension(
    filename
):

    if not filename:

        raise ValueError(
            "Image filename is missing."
        )


    extension = (
        Path(
            filename
        )
        .suffix
        .lower()
    )


    if extension not in ALLOWED_EXTENSIONS:

        raise ValueError(
            "Unsupported image type. "
            "Only PNG, JPG and JPEG are allowed."
        )


    return extension


# ============================================================
# VERIFY REAL IMAGE CONTENT
#
# File extensions alone are not trusted.
#
# An attacker could rename:
#
# malicious.exe
#
# to:
#
# chest_xray.jpg
#
# PIL verifies that the bytes are actually an image.
# ============================================================

def verify_image_file(
    path
):

    path = Path(
        path
    )


    if not path.exists():

        raise FileNotFoundError(
            f"Image does not exist: {path}"
        )


    if not path.is_file():

        raise ValueError(
            "Image path is not a regular file."
        )


    file_size = (
        path.stat()
        .st_size
    )


    if file_size <= 0:

        raise ValueError(
            "Uploaded image is empty."
        )


    if (
        file_size
        >
        MAX_UPLOAD_BYTES
    ):

        raise ValueError(
            "Uploaded image exceeds "
            "the 15 MB safety limit."
        )


    try:

        with warnings.catch_warnings():

            warnings.simplefilter(
                "error",
                Image.DecompressionBombWarning,
            )


            # ------------------------------------------------
            # FIRST PASS
            #
            # verify() checks image structure without fully
            # decoding every pixel.
            # ------------------------------------------------

            with Image.open(
                path
            ) as image:

                image_format = (
                    image.format
                )


                width, height = (
                    image.size
                )


                frame_count = getattr(
                    image,
                    "n_frames",
                    1,
                )


                image.verify()


        # ----------------------------------------------------
        # SECOND PASS
        #
        # Re-open because verify() invalidates the first
        # Image object.
        # ----------------------------------------------------

        with warnings.catch_warnings():

            warnings.simplefilter(
                "error",
                Image.DecompressionBombWarning,
            )


            with Image.open(
                path
            ) as image:

                image.load()


    except (
        UnidentifiedImageError,
        OSError,
        SyntaxError,
        Image.DecompressionBombWarning,
        Image.DecompressionBombError,
    ) as exc:

        raise ValueError(
            "The uploaded file is not a safe "
            "supported image."
        ) from exc


    if image_format not in ALLOWED_FORMATS:

        raise ValueError(
            "Unsupported image format. "
            "Only PNG and JPEG images are accepted."
        )


    if (
        width <= 0
        or
        height <= 0
    ):

        raise ValueError(
            "Image dimensions are invalid."
        )


    if (
        width
        >
        MAX_IMAGE_WIDTH
        or
        height
        >
        MAX_IMAGE_HEIGHT
    ):

        raise ValueError(

            "Image dimensions are too large. "

            f"Maximum supported size is "
            f"{MAX_IMAGE_WIDTH} x "
            f"{MAX_IMAGE_HEIGHT} pixels."
        )


    if frame_count != 1:

        raise ValueError(
            "Animated or multi-frame images "
            "are not accepted."
        )


    # ========================================================
    # VERIFY EXTENSION MATCHES ACTUAL CONTENT
    # ========================================================

    extension = (
        path.suffix.lower()
    )


    if (
        image_format
        ==
        "PNG"
        and
        extension
        !=
        ".png"
    ):

        raise ValueError(
            "Image extension does not match "
            "the actual PNG content."
        )


    if (
        image_format
        ==
        "JPEG"
        and
        extension
        not in {
            ".jpg",
            ".jpeg",
        }
    ):

        raise ValueError(
            "Image extension does not match "
            "the actual JPEG content."
        )


    return {

        "format":
            image_format,

        "width":
            int(
                width
            ),

        "height":
            int(
                height
            ),

        "size_bytes":
            int(
                file_size
            ),
    }


# ============================================================
# SHA-256 FILE HASH
#
# Useful later for:
#
# - integrity verification
# - detecting accidental modifications
# - audit records
# ============================================================

def calculate_sha256(
    path
):

    path = Path(
        path
    )


    digest = hashlib.sha256()


    with path.open(
        "rb"
    ) as file_handle:

        while True:

            chunk = file_handle.read(
                1024
                *
                1024
            )


            if not chunk:

                break


            digest.update(
                chunk
            )


    return digest.hexdigest()


# ============================================================
# FLUSH DIRECTORY METADATA
#
# After os.replace(), flushing the directory reduces the risk
# that an unexpected power loss leaves the rename only partly
# persisted on Linux.
# ============================================================

def _sync_directory(
    directory
):

    try:

        descriptor = os.open(

            str(
                directory
            ),

            os.O_RDONLY,
        )


        try:

            os.fsync(
                descriptor
            )


        finally:

            os.close(
                descriptor
            )


    except OSError:

        pass


# ============================================================
# ATOMIC UPLOAD SAVE
#
# Instead of:
#
# upload -> final patient file directly
#
# we use:
#
# upload
#    ↓
# temporary file
#    ↓
# validate image
#    ↓
# fsync
#    ↓
# atomic rename
#    ↓
# final stored image
#
# A failed upload therefore does not leave a half-written
# patient image in screenings/originals.
# ============================================================

def save_uploaded_image(
    uploaded_file,
    destination_path
):

    initialise_storage()


    destination_path = (
        require_original_path(
            destination_path
        )
    )


    validate_extension(
        destination_path.name
    )


    temporary_path = (

        TEMP_DIR
        /
        (
            destination_path.name
            +
            ".uploading"
        )
    )


    # ========================================================
    # REMOVE OLD TEMP FILE IF ONE EXISTS
    # ========================================================

    if temporary_path.exists():

        try:

            temporary_path.unlink()

        except OSError as exc:

            raise RuntimeError(
                "Could not clear temporary "
                "upload file."
            ) from exc


    total_bytes = 0


    try:

        # ====================================================
        # WRITE TEMPORARY FILE
        # ====================================================

        with temporary_path.open(
            "wb"
        ) as output:

            while True:

                chunk = (
                    uploaded_file.stream.read(
                        1024
                        *
                        1024
                    )
                )


                if not chunk:

                    break


                total_bytes += len(
                    chunk
                )


                if (
                    total_bytes
                    >
                    MAX_UPLOAD_BYTES
                ):

                    raise ValueError(
                        "Uploaded image exceeds "
                        "the 15 MB safety limit."
                    )


                output.write(
                    chunk
                )


            output.flush()


            os.fsync(
                output.fileno()
            )


        # ====================================================
        # TEMP FILE NEEDS THE REAL EXTENSION FOR VALIDATION
        # ====================================================

        verification_path = (

            TEMP_DIR
            /
            (
                destination_path.stem
                +
                destination_path.suffix
                +
                ".verify"
                +
                destination_path.suffix
            )
        )


        if verification_path.exists():

            verification_path.unlink()


        os.replace(

            temporary_path,

            verification_path,
        )


        try:

            metadata = verify_image_file(
                verification_path
            )


            file_hash = calculate_sha256(
                verification_path
            )


            # ================================================
            # FINAL ATOMIC RENAME
            # ================================================

            os.replace(

                verification_path,

                destination_path,
            )


            try:

                os.chmod(
                    destination_path,
                    0o600,
                )

            except OSError:

                pass


            _sync_directory(
                ORIGINALS_DIR
            )


        finally:

            if verification_path.exists():

                verification_path.unlink()


        return {

            "path":
                str(
                    destination_path
                ),

            "sha256":
                file_hash,

            "size_bytes":
                metadata[
                    "size_bytes"
                ],

            "width":
                metadata[
                    "width"
                ],

            "height":
                metadata[
                    "height"
                ],

            "format":
                metadata[
                    "format"
                ],
        }


    finally:

        if temporary_path.exists():

            try:

                temporary_path.unlink()

            except OSError:

                pass


# ============================================================
# VALIDATE GRAD-CAM AFTER ENGINE CREATES IT
# ============================================================

def validate_gradcam_file(
    path
):

    path = (
        require_gradcam_path(
            path
        )
    )


    metadata = verify_image_file(
        path
    )


    file_hash = calculate_sha256(
        path
    )


    try:

        os.chmod(
            path,
            0o600,
        )

    except OSError:

        pass


    return {

        "path":
            str(
                path
            ),

        "sha256":
            file_hash,

        "size_bytes":
            metadata[
                "size_bytes"
            ],

        "width":
            metadata[
                "width"
            ],

        "height":
            metadata[
                "height"
            ],

        "format":
            metadata[
                "format"
            ],
    }


# ============================================================
# SAFE FILE EXISTENCE CHECK
# ============================================================

def managed_file_exists(
    path
):

    if not path:

        return False


    try:

        path = (
            require_managed_path(
                path
            )
        )


    except (
        ValueError,
        OSError
    ):

        return False


    return (
        path.exists()
        and
        path.is_file()
    )


# ============================================================
# SAFE DELETE
#
# Only TB Guard managed files can be deleted.
#
# A database value such as:
#
# /etc/passwd
#
# can therefore never cause this function to delete an
# operating-system file.
# ============================================================

def delete_managed_file(
    path
):

    if not path:

        return False


    try:

        path = (
            require_managed_path(
                path
            )
        )


    except (
        ValueError,
        OSError
    ):

        raise ValueError(
            "Refusing to delete a file outside "
            "TB Guard managed storage."
        )


    if not path.exists():

        return False


    if not path.is_file():

        raise ValueError(
            "Managed path is not a regular file."
        )


    path.unlink()


    _sync_directory(
        path.parent
    )


    return True


# ============================================================
# STORAGE AUDIT
#
# Compares database paths with files on disk.
#
# Returns:
#
# missing_database_files
#     Database points to a file that is missing.
#
# orphan_originals
#     Original X-ray exists but database does not reference it.
#
# orphan_gradcams
#     Grad-CAM exists but database does not reference it.
# ============================================================

def audit_storage(
    screening_rows
):

    initialise_storage()


    database_originals = set()

    database_gradcams = set()

    missing_database_files = []


    for row in screening_rows:

        row = dict(
            row
        )


        screening_id = row.get(
            "screening_id"
        )


        image_path = row.get(
            "image_path"
        )


        gradcam_path = row.get(
            "gradcam_path"
        )


        # ====================================================
        # ORIGINAL IMAGE
        # ====================================================

        if image_path:

            try:

                safe_image_path = (
                    require_original_path(
                        image_path
                    )
                )


                database_originals.add(
                    safe_image_path
                )


                if not safe_image_path.exists():

                    missing_database_files.append(
                        {
                            "screening_id":
                                screening_id,

                            "type":
                                "original",

                            "path":
                                str(
                                    safe_image_path
                                ),
                        }
                    )


            except ValueError:

                missing_database_files.append(
                    {
                        "screening_id":
                            screening_id,

                        "type":
                            "unsafe_original_path",

                        "path":
                            str(
                                image_path
                            ),
                    }
                )


        # ====================================================
        # GRAD-CAM
        # ====================================================

        if gradcam_path:

            try:

                safe_gradcam_path = (
                    require_gradcam_path(
                        gradcam_path
                    )
                )


                database_gradcams.add(
                    safe_gradcam_path
                )


                if not safe_gradcam_path.exists():

                    missing_database_files.append(
                        {
                            "screening_id":
                                screening_id,

                            "type":
                                "gradcam",

                            "path":
                                str(
                                    safe_gradcam_path
                                ),
                        }
                    )


            except ValueError:

                missing_database_files.append(
                    {
                        "screening_id":
                            screening_id,

                        "type":
                            "unsafe_gradcam_path",

                        "path":
                            str(
                                gradcam_path
                            ),
                    }
                )


    # ========================================================
    # FILES ACTUALLY ON DISK
    # ========================================================

    disk_originals = {

        path.resolve()

        for path
        in ORIGINALS_DIR.iterdir()

        if path.is_file()
    }


    disk_gradcams = {

        path.resolve()

        for path
        in GRADCAM_DIR.iterdir()

        if path.is_file()
    }


    orphan_originals = sorted(

        str(
            path
        )

        for path
        in (
            disk_originals
            -
            database_originals
        )
    )


    orphan_gradcams = sorted(

        str(
            path
        )

        for path
        in (
            disk_gradcams
            -
            database_gradcams
        )
    )


    return {

        "missing_database_files":
            missing_database_files,

        "orphan_originals":
            orphan_originals,

        "orphan_gradcams":
            orphan_gradcams,
    }


# ============================================================
# CLEAN TEMPORARY UPLOAD FILES
#
# Useful after a power failure or interrupted upload.
# ============================================================

def clean_temporary_files():

    initialise_storage()


    removed = 0


    for path in TEMP_DIR.iterdir():

        if not path.is_file():

            continue


        try:

            path.unlink()

            removed += 1


        except OSError:

            pass


    return removed


# ============================================================
# STARTUP
# ============================================================

initialise_storage()
