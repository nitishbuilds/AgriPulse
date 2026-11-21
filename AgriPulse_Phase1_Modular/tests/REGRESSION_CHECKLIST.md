# Regression test checklist — Phase 1
These are planned checks, not claimed as passed.

| ID | Area | Test | Expected result |
|---|---|---|---|
| TC-01 | App startup | Run app.py in Colab | Streamlit starts without import/syntax error |
| TC-02 | Home UI | Open home page | AgriPulse branding and navigation visible |
| TC-03 | Contact | Open Contact Us and submit blank form | Validation warning appears |
| TC-04 | Contact | Submit completed form | Existing validation success message appears; no email claim |
| TC-05 | Weather | Select district and refresh | Weather panel renders or reports provider/network error |
| TC-06 | Mandi | Load bundled CSV | Table renders with CSV source identified |
| TC-07 | AI | Ask question with valid OpenRouter key | Response appears |
| TC-08 | ML | Load model and labels | Model loads with expected 38 output labels |
| TC-09 | ML | Upload supported potato/maize/tomato image | Existing prediction path runs |
| TC-10 | Transport | Enter sample distance and rates | Cost formula output matches entered values |
| TC-11 | Navigation | Open and close dialogs | Other app panels remain usable |
| TC-12 | Theme | Toggle light/dark mode | Theme toggle still works |
