"""Google Colab launcher for the modular AgriPulse project."""
from __future__ import annotations

import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

PROJECT_DIR = pathlib.Path(__file__).resolve().parent
DRIVE_BACKUP = pathlib.Path("/content/drive/MyDrive/AgriPulse_Backup")
PERSISTENT_DATA = pathlib.Path("/content/drive/MyDrive/AgriPulse_Data")
PERSISTENT_DATA.mkdir(parents=True, exist_ok=True)

# Persist SQLite outside the temporary Colab runtime.
os.environ["AGRIPULSE_DB_PATH"] = str(PERSISTENT_DATA / "agripulse.db")

# Restore the existing trained model and labels from the user's Drive backup.
for filename in ("agripulse_plant_disease.keras", "agripulse_labels.json"):
    source = DRIVE_BACKUP / filename
    destination = PROJECT_DIR / "models" / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        raise FileNotFoundError(
            f"Missing {source}. Upload the existing model and labels to "
            "Google Drive > MyDrive > AgriPulse_Backup."
        )
    shutil.copy2(source, destination)

# The app reads its mandi CSV directly from PROJECT_DIR/data.

# Stop only previous AgriPulse processes, then launch Streamlit.
subprocess.run("pkill -f '[s]treamlit run .*AgriPulse.*/app.py' || true",
               shell=True, check=False)
subprocess.run("fuser -k 8501/tcp >/dev/null 2>&1 || true",
               shell=True, check=False)

log_file = open("/content/streamlit.log", "w", encoding="utf-8")
subprocess.Popen(
    [sys.executable, "-m", "streamlit", "run", str(PROJECT_DIR / "app.py"),
     "--server.port", "8501", "--server.address", "0.0.0.0",
     "--server.headless", "true"],
    cwd=str(PROJECT_DIR), stdout=log_file, stderr=subprocess.STDOUT,
)

# Install Cloudflare Tunnel if it is not already present.
cloudflared = pathlib.Path("/usr/local/bin/cloudflared")
if not cloudflared.exists():
    subprocess.run(
        "wget -q https://github.com/cloudflare/cloudflared/"
        "releases/latest/download/cloudflared-linux-amd64 "
        "-O /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared",
        shell=True, check=True,
    )

subprocess.Popen(
    '"/usr/local/bin/cloudflared" tunnel --url http://127.0.0.1:8501 '
    '--no-autoupdate > /content/cloudflare.log 2>&1',
    shell=True,
)
print("AgriPulse starting. Wait for the Cloudflare URL below.")
for _ in range(30):
    time.sleep(2)
    log_path = pathlib.Path("/content/cloudflare.log")
    if log_path.exists():
        matches = re.findall(
            r"https://[A-Za-z0-9-]+[.]trycloudflare[.]com",
            log_path.read_text(errors="ignore"),
        )
        if matches:
            print("AgriPulse URL:", matches[-1])
            print("SQLite database:", os.environ["AGRIPULSE_DB_PATH"])
            break
else:
    print("Tunnel URL not found. Inspect /content/streamlit.log and cloudflare.log.")
