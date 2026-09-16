import sqlite3
import hashlib
from pathlib import Path
from datetime import datetime

from database import (
    backup_database,
    DATABASE_PATH,
)

EVIDENCE_FILE = Path(
    "research_evidence/09_database_backup_recovery.txt"
)

BACKUP_PATH = Path(
    "backups/point9_verification_backup.db"
)

RESTORE_PATH = Path(
    "backups/point9_restored_test.db"
)


def quick_check(path):
    connection = sqlite3.connect(str(path))

    try:
        result = connection.execute(
            "PRAGMA quick_check"
        ).fetchone()

        return result[0] if result else None

    finally:
        connection.close()


def table_counts(path):
    connection = sqlite3.connect(str(path))

    try:
        tables = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        ).fetchall()

        counts = {}

        for (table_name,) in tables:
            safe_name = table_name.replace('"', '""')

            count = connection.execute(
                f'SELECT COUNT(*) FROM "{safe_name}"'
            ).fetchone()[0]

            counts[table_name] = count

        return counts

    finally:
        connection.close()


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


# ============================================================
# CREATE VERIFIED BACKUP
# ============================================================

backup_database(
    BACKUP_PATH
)


# ============================================================
# RESTORE BACKUP INTO A SEPARATE TEST DATABASE
# ============================================================

if RESTORE_PATH.exists():
    RESTORE_PATH.unlink()

source = sqlite3.connect(
    str(BACKUP_PATH)
)

restored = sqlite3.connect(
    str(RESTORE_PATH)
)

try:
    source.backup(
        restored
    )

    restored.commit()

finally:
    restored.close()
    source.close()


# ============================================================
# VALIDATE LIVE, BACKUP AND RESTORED DATABASES
# ============================================================

live_integrity = quick_check(
    DATABASE_PATH
)

backup_integrity = quick_check(
    BACKUP_PATH
)

restore_integrity = quick_check(
    RESTORE_PATH
)

live_counts = table_counts(
    DATABASE_PATH
)

backup_counts = table_counts(
    BACKUP_PATH
)

restore_counts = table_counts(
    RESTORE_PATH
)

backup_matches_live = (
    backup_counts == live_counts
)

restore_matches_backup = (
    restore_counts == backup_counts
)

backup_hash = sha256_file(
    BACKUP_PATH
)

restore_hash = sha256_file(
    RESTORE_PATH
)


# ============================================================
# SAVE EVIDENCE
# ============================================================

lines = []

lines.append(
    "======================================"
)

lines.append(
    "TB GUARD - SQLITE BACKUP/RECOVERY TEST"
)

lines.append(
    "======================================"
)

lines.append("")

lines.append(
    f"Test completed: {datetime.now().isoformat()}"
)

lines.append("")

lines.append(
    f"Live database: {DATABASE_PATH}"
)

lines.append(
    f"Backup database: {BACKUP_PATH}"
)

lines.append(
    f"Restored test database: {RESTORE_PATH}"
)

lines.append("")

lines.append(
    f"Live integrity: {live_integrity}"
)

lines.append(
    f"Backup integrity: {backup_integrity}"
)

lines.append(
    f"Restored integrity: {restore_integrity}"
)

lines.append("")

lines.append(
    f"Backup table counts match live DB: "
    f"{backup_matches_live}"
)

lines.append(
    f"Restored table counts match backup: "
    f"{restore_matches_backup}"
)

lines.append("")

lines.append(
    "===== TABLE COUNTS ====="
)

for table_name, count in live_counts.items():

    lines.append(
        f"{table_name}: {count}"
    )

lines.append("")

lines.append(
    f"Backup SHA-256: {backup_hash}"
)

lines.append(
    f"Restored SHA-256: {restore_hash}"
)

lines.append("")

overall_pass = all([
    str(live_integrity).lower() == "ok",
    str(backup_integrity).lower() == "ok",
    str(restore_integrity).lower() == "ok",
    backup_matches_live,
    restore_matches_backup,
])

lines.append(
    "FINAL RESULT: "
    + (
        "PASS"
        if overall_pass
        else "FAIL"
    )
)

EVIDENCE_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

EVIDENCE_FILE.write_text(
    "\n".join(lines) + "\n"
)

print(
    "\n".join(lines)
)
