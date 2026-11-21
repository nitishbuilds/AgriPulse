> **Streamlit Cloud users:** Read [STREAMLIT_DEPLOY.md](STREAMLIT_DEPLOY.md) first. The existing deployed app uses the `AgriPulse_Phase1_Modular/` folder, and this copy keeps that path. The Colab instructions below are historical.

# AgriPulse — SQLite authentication build

## What's included
- `app.py`: intentionally small Streamlit entry point.
- `pages/home.py`: original AgriPulse app code, preserved with an authentication gate and account controls.
- `database/auth.py`: SQLite schema, registration, login, password hashing and guest access.
- `data/agripulse_mandi_prices.csv`: original embedded mandi snapshot extracted to a separate asset.
- `backups/original_app_code.py`: exact original embedded Streamlit app code before auth changes.
- `colab_launcher.py`: Colab + Google Drive + Streamlit + Cloudflare launcher.
- `tests/`: authentication tests and regression test plan.

## Google Colab
1. Upload and extract this project ZIP into `/content/AgriPulse_Phase1_Modular`.
2. Mount Google Drive in a Colab cell:
   `from google.colab import drive; drive.mount('/content/drive')`
3. Ensure the existing model and labels are in:
   `MyDrive/AgriPulse_Backup/agripulse_plant_disease.keras`
   `MyDrive/AgriPulse_Backup/agripulse_labels.json`
4. Install requirements: `%pip install -r /content/AgriPulse_Phase1_Modular/requirements.txt`
5. Run `colab_launcher.py`.
6. SQLite database is stored at `MyDrive/AgriPulse_Data/agripulse.db`.

## Local run
Install requirements, set `AGRIPULSE_DB_PATH` to a persistent local path, then run:
`streamlit run app.py`

## Security / limitations
- Passwords are PBKDF2-HMAC-SHA256 hashes with random salts; plaintext passwords are never stored.
- SQL statements use placeholders.
- Guest mode does not create a user account.
- Password reset/email verification and admin account provisioning are not included yet.
- The existing app logic is retained in `pages/home.py`; further component-by-component extraction is a separate, test-after-each-change refactor.
- Colab's `/content` is temporary. The launcher stores the SQLite DB in Google Drive.
- Do not commit the SQLite database, credentials, or API keys to a public repository.
