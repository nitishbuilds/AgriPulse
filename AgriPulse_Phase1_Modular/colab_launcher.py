# Google Colab launcher for modular AgriPulse
# Upload/extract the AgriPulse_Phase1_Modular.zip, then run this cell.
from google.colab import drive
drive.mount("/content/drive")

import os, sys, shutil, subprocess, time, pathlib, re
PROJECT_ZIP = "/content/AgriPulse_Phase1_Modular.zip"  # change if uploaded elsewhere
PROJECT_DIR = "/content/AgriPulse_Phase1_Modular"

if not os.path.exists(PROJECT_ZIP):
    raise FileNotFoundError(f"Upload the project ZIP to /content first: {PROJECT_ZIP}")

shutil.rmtree(PROJECT_DIR, ignore_errors=True)
subprocess.run(["unzip", "-q", PROJECT_ZIP, "-d", "/content"], check=True)

# Preserve current Colab model/labels locations expected by the original ML code.
# Copy from the user's Drive backup if present; otherwise the app will show model unavailable.
drive_candidates = [
    "/content/drive/MyDrive/AgriPulse_Backup/agripulse_plant_disease.keras",
    "/content/drive/MyDrive/AgriPulse_Backup/agripulse_labels.json",
]
for src in drive_candidates:
    if os.path.exists(src):
        shutil.copy2(src, "/content/" + os.path.basename(src))

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                os.path.join(PROJECT_DIR, "requirements.txt")], check=True)

subprocess.Popen([sys.executable, "-m", "streamlit", "run",
                  os.path.join(PROJECT_DIR, "app.py"),
                  "--server.port", "8501", "--server.address", "0.0.0.0",
                  "--server.headless", "true"])
time.sleep(5)
# Start Cloudflare tunnel separately if cloudflared is installed in this runtime.
subprocess.Popen("cloudflared tunnel --url http://127.0.0.1:8501 --no-autoupdate",
                 shell=True)
