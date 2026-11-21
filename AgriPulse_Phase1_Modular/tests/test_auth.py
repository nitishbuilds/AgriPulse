import importlib.util
import os
import sys
from pathlib import Path

def load_auth_module(tmp_path, monkeypatch):
    module_path = Path(__file__).resolve().parents[1] / "database" / "auth.py"
    monkeypatch.setenv("AGRIPULSE_DB_PATH", str(tmp_path / "test.db"))
    spec = importlib.util.spec_from_file_location("auth_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.initialize_database()
    return module

def test_registration_and_login(tmp_path, monkeypatch):
    auth = load_auth_module(tmp_path, monkeypatch)
    ok, _ = auth.register_user("Test Farmer", "farmer@example.com", "strongpass123")
    assert ok
    user = auth.authenticate_user("farmer@example.com", "strongpass123")
    assert user is not None
    assert user["role"] == "farmer"

def test_duplicate_email_rejected(tmp_path, monkeypatch):
    auth = load_auth_module(tmp_path, monkeypatch)
    assert auth.register_user("Test Farmer", "farmer@example.com", "strongpass123")[0]
    assert not auth.register_user("Other Farmer", "FARMER@example.com", "strongpass123")[0]

def test_wrong_password_rejected(tmp_path, monkeypatch):
    auth = load_auth_module(tmp_path, monkeypatch)
    auth.register_user("Test Farmer", "farmer@example.com", "strongpass123")
    assert auth.authenticate_user("farmer@example.com", "wrongpass") is None

def test_registration_validation(tmp_path, monkeypatch):
    auth = load_auth_module(tmp_path, monkeypatch)
    assert auth.validate_registration("A", "bad-email", "short", "no-match") is not None
    assert auth.validate_registration("Valid Farmer", "valid@example.com",
                                      "strongpass123", "strongpass123") is None
