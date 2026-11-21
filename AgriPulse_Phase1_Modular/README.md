# AgriPulse — Phase 1 modular structure

## What was done
- Preserved the supplied AgriPulse Python launcher in `backups/`.
- Extracted the embedded Streamlit application into `pages/home.py`.
- Added a stable root `app.py` entrypoint.
- Extracted the original CSS into `assets/styles/main.css` and loads it at runtime.
- Extracted the Contact Us dialog into `components/contact.py` while preserving its current behaviour.
- Extracted the bundled mandi CSV snapshot into `data/mandi_snapshot.csv`.
- Added a Google Colab launcher and dependency file.

## Important preservation notes
- No ML model, training, confidence threshold, OpenRouter model, API behaviour, weather logic, mandi logic, transport formula, or UI feature was intentionally redesigned in Phase 1.
- The original source file is preserved byte-for-byte in `backups/`.
- The existing ML model and labels must still be present in Google Drive at:
  - `AgriPulse_Backup/agripulse_plant_disease.keras`
  - `AgriPulse_Backup/agripulse_labels.json`
- The current Contact Us behaviour is unchanged: it validates the form but does not send email yet.
- API key handling remains as in the original app for now; moving it to Colab Secrets is a later phase.
- Session-based chat history, improved leaf rejection, tests, and animations are later phases.

## How to run in Google Colab
1. Upload this ZIP to `/content/` in Colab.
2. Run `colab_launcher.py` in a notebook cell (or copy its contents).
3. Ensure the model and labels are in the stated Drive folder.
4. Confirm the Streamlit URL appears and test every existing feature.

## Important
This is a first modular migration candidate, not yet a verified replacement. Keep the old launcher until the regression checklist passes. Do not delete your old notebook.
