# Optional ML artifacts

`agripulse_plant_disease.keras` and `agripulse_labels.json` are **not included** in the uploaded ZIP. Disease predictions remain unavailable until both are supplied.

Put both files here, or set the environment variables `AGRIPULSE_MODEL_PATH` and `AGRIPULSE_LABELS_PATH` to existing paths. Files from the original Google Drive Colab environment are not automatically present on Streamlit Cloud.

Do not commit data containing credentials to this directory. For large models, use a suitable external model store and modify the loader accordingly.
