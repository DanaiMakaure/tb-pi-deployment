============================================================
TB GUARD - OFFLINE RECOVERY GUIDE
============================================================

Purpose
-------
This recovery package contains the files required to rebuild
the TB Guard Raspberry Pi screening prototype without requiring
Internet access for Python package installation.

Validated deployment environment
--------------------------------
Device:
Raspberry Pi 400 Rev 1.0

Architecture:
armv7l

Operating System:
Raspbian GNU/Linux 11 (Bullseye)

Python:
3.9.2

TensorFlow Lite Runtime:
2.11.0


============================================================
1. CREATE PROJECT DIRECTORY
============================================================

mkdir -p /home/kingly/tb-pi-deployment

cd /home/kingly/tb-pi-deployment


============================================================
2. COPY APPLICATION FILES
============================================================

Copy the contents of:

application/

into:

/home/kingly/tb-pi-deployment/


The project should contain:

app.py
database.py
screening_engine.py
storage.py
templates/


============================================================
3. COPY MODEL FILES
============================================================

Copy all files from:

models/

into:

/home/kingly/tb-pi-deployment/


Required model artifacts:

mobilenetv3_tb_pi.tflite
mobilenetv3_features_pi.tflite
mobilenetv3_gradcam_head.npz


============================================================
4. CREATE PYTHON VIRTUAL ENVIRONMENT
============================================================

cd /home/kingly/tb-pi-deployment

python3 -m venv .venv

source .venv/bin/activate


============================================================
5. INSTALL DEPENDENCIES OFFLINE
============================================================

The offline_packages directory must be available locally.

Run:

python -m pip install \
--no-index \
--find-links=offline_packages \
-r requirements.txt


No Internet package index is required.


============================================================
6. TEST REQUIRED PYTHON DEPENDENCIES
============================================================

python - <<'PY'
import flask
import numpy
import PIL
import tflite_runtime.interpreter

print("Flask:", flask.__version__)
print("NumPy:", numpy.__version__)
print("Pillow:", PIL.__version__)
print("TFLite Runtime: IMPORT SUCCESS")
PY


============================================================
7. TEST TB GUARD MANUALLY
============================================================

python app.py

Then open:

http://127.0.0.1:5000/

or from another device on the same local network:

http://<RASPBERRY_PI_IP>:5000/


============================================================
8. RESTORE AUTOMATIC STARTUP
============================================================

Copy:

systemd/tb-guard.service

to:

/etc/systemd/system/tb-guard.service


Run:

sudo systemctl daemon-reload

sudo systemctl enable tb-guard.service

sudo systemctl start tb-guard.service


Verify:

systemctl is-enabled tb-guard.service

systemctl is-active tb-guard.service


Expected:

enabled
active


============================================================
9. VERIFY LOCAL WEB APPLICATION
============================================================

curl -s -o /dev/null \
-w "HTTP %{http_code}\n" \
http://127.0.0.1:5000/


Expected:

HTTP 200


============================================================
10. DATABASE RECOVERY
============================================================

Patient/screening databases are intentionally NOT included in
this general recovery package.

If an authorised verified SQLite backup is available, restore it
separately as:

/home/kingly/tb-pi-deployment/screening.db

Before using a restored database, verify it with:

sqlite3 screening.db "PRAGMA quick_check;"


Expected:

ok


============================================================
11. IMPORTANT SECURITY / DATA NOTE
============================================================

The general recovery package intentionally excludes:

screening.db
screenings/originals/
screenings/gradcam/
uploads/
gradcam_test_images/
database backup files containing screening records

These must be backed up and handled separately according to the
research project's data protection requirements.


============================================================
12. SYSTEMD PATH ASSUMPTION
============================================================

The included tb-guard.service assumes:

User:
kingly

Project directory:
/home/kingly/tb-pi-deployment

Python:
/home/kingly/tb-pi-deployment/.venv/bin/python

If the Raspberry Pi username or installation directory changes,
edit tb-guard.service before enabling it.


============================================================
END OF RECOVERY GUIDE
============================================================
