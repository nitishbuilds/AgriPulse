# AgriPulse regression test plan

Run automated authentication tests with `pytest -q tests/test_auth.py`.

| ID | Scenario | Expected result |
|---|---|---|
| AUTH-01 | Register with valid details | Account and farmer profile are created |
| AUTH-02 | Register same email again | Duplicate account is rejected |
| AUTH-03 | Register with invalid email | Validation message is shown |
| AUTH-04 | Register with password under 8 chars | Validation message is shown |
| AUTH-05 | Register with mismatched passwords | Validation message is shown |
| AUTH-06 | Login with correct credentials | Farmer enters existing AgriPulse app |
| AUTH-07 | Login with incorrect password | Access is denied |
| AUTH-08 | Continue as guest | Existing app opens without account |
| AUTH-09 | Logout | Session returns to login/guest screen |
| AUTH-10 | Restart Colab runtime with Drive mounted | SQLite accounts remain available |
| REG-01 | Existing crop prediction flow | Same model and prediction code run |
| REG-02 | Existing weather and mandi features | Existing UI and API behaviour retained |
| REG-03 | Existing transport planner | Existing calculation and dialog remain |
| REG-04 | Existing OpenRouter flow | Existing integration remains |
| REG-05 | Existing CSV snapshot | Same bundled market data remains available |

## Manual checks still required
- Run the app in the user's Colab runtime.
- Confirm Google Drive mount and model restoration.
- Test each existing page, button, dialog and API with real credentials.
- Verify SQLite database path points to Google Drive in Colab.
- Verify Cloudflare URL opens the same application.
- Do not mark manual tests PASS until actually executed.
