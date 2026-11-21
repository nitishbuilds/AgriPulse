# AgriPulse SQLite Authentication — Streamlit Community Cloud

This is a deployment-prepared copy of `AgriPulse_SQLite_Auth (1).zip`.

## Why the directory name differs

The current live Streamlit Cloud app points to `AgriPulse_Phase1_Modular/app.py`. This package intentionally keeps that directory and entrypoint path so your existing app URL and Streamlit deployment remain the same. The code inside is from **AgriPulse_SQLite_Auth** (new build).

## Publish with GitHub Desktop (recommended)

1. Clone the existing `nitishbuilds/AgriPulse` repository with GitHub Desktop. If available, create a backup branch or ZIP of the current repo before replacing files.
2. In the cloned repository, **delete the old `AgriPulse_Phase1_Modular` directory** locally. This matters: uploading newer files through GitHub's web UI does **not** delete files from the older version.
3. Copy the `AgriPulse_Phase1_Modular` directory from this package into the cloned repository root, ensuring there is no extra enclosing `AgriPulse_Latest_Streamlit_Ready/` folder in the repository.
4. In GitHub Desktop, check the full diff for accidental changes, commit `Upgrade AgriPulse to SQLite auth version`, then **Push origin**.
5. Existing Streamlit app updates automatically from GitHub. No need to create a new app or change the entrypoint path.

Expected project path: `AgriPulse/AgriPulse_Phase1_Modular/app.py`.

## What was changed for hosting

- Mandi CSV loads from `data/agripulse_mandi_prices.csv`, not `/content/`.
- Model/labels load from optional `models/` files, or from `AGRIPULSE_MODEL_PATH` and `AGRIPULSE_LABELS_PATH` environment variables.
- TensorFlow import occurs when a model is actually available, to reduce initial load.
- `pytest` was removed from production requirements (tests are retained and can be run with `pip install pytest`).
- Colab launcher copies its model files into local `models/` to match the updated loader.
- Local SQLite signups carry a demo-data persistence warning.
- Generated `__pycache__`, `.pytest_cache`, and historical original-code backups are intentionally omitted from this **deployment** copy. Your original uploaded ZIP remains the full archive.

## Limitations to fix before public production use

1. **SQLite is NOT reliable permanent storage on Streamlit Community Cloud.** Any registered farmer accounts can be lost on app reboot/redeploy/container recycle. Do not onboard real users yet. Switch to managed persistent PostgreSQL (or another reliable cloud DB) for production.
2. The trained `.keras` model and label JSON are **not included**; ML prediction says model unavailable until you add both and verify inference compatibility.
3. User-entered OpenRouter and data.gov.in API keys are entered in the UI. Don't commit credentials; if the app owner supplies keys, use Streamlit's Secrets settings.
4. TensorFlow is a large dependency; build/RAM compatibility must be confirmed from your actual Streamlit logs.
5. Review authentication rate limiting, email verification, password recovery, password policy, user privacy and session revocation before using public real accounts.

## Local smoke test

From repository root:

```bash
pip install -r AgriPulse_Phase1_Modular/requirements.txt
streamlit run AgriPulse_Phase1_Modular/app.py
```

Authentication tests (requires pytest separately):

```bash
pip install pytest
python -m pytest -q AgriPulse_Phase1_Modular/tests/test_auth.py
```

## If you choose the browser upload method

Upload files from inside this package's `AgriPulse_Phase1_Modular` folder into **the same folder** in GitHub so the paths match. However, GitHub's browser UI won't cleanly remove obsolete folders and files. Use GitHub Desktop if you truly want to replace the old version.
