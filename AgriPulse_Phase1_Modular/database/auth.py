"""SQLite-backed farmer authentication for AgriPulse.

Passwords are stored as PBKDF2-HMAC-SHA256 hashes with per-user random salts.
The database path can be overridden with AGRIPULSE_DB_PATH.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = PROJECT_ROOT / "data" / "agripulse.db"
DB_PATH = Path(os.environ.get("AGRIPULSE_DB_PATH", str(DEFAULT_DB)))
ITERATIONS = 310_000


def get_connection() -> sqlite3.Connection:
    """Open a SQLite connection and return rows by column name."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(DB_PATH), timeout=20)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


def initialize_database() -> None:
    """Create authentication and farmer-profile tables if they do not exist."""
    with get_connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'farmer'
                    CHECK (role IN ('farmer', 'admin')),
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                is_active INTEGER NOT NULL DEFAULT 1
                    CHECK (is_active IN (0, 1))
            );

            CREATE TABLE IF NOT EXISTS farmer_profiles (
                user_id INTEGER PRIMARY KEY,
                full_name TEXT NOT NULL,
                phone TEXT,
                village TEXT,
                district TEXT,
                state TEXT DEFAULT 'Madhya Pradesh',
                primary_crop TEXT,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS activity_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                event_type TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
            );
            """
        )


def _hash_password(password: str, salt_hex: str | None = None) -> tuple[str, str]:
    """Hash a password with PBKDF2 and return (hash, salt) as hex strings."""
    salt = bytes.fromhex(salt_hex) if salt_hex else secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, ITERATIONS
    )
    return digest.hex(), salt.hex()


def validate_registration(full_name: str, email: str, password: str,
                          confirm_password: str) -> str | None:
    """Return a user-friendly validation message, or None when valid."""
    if len(full_name.strip()) < 2:
        return "Please enter your full name."
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email.strip()):
        return "Please enter a valid email address."
    if len(password) < 8:
        return "Password must contain at least 8 characters."
    if password != confirm_password:
        return "Passwords do not match."
    return None


def register_user(full_name: str, email: str, password: str) -> tuple[bool, str]:
    """Create a farmer account and its initial profile."""
    normalized_email = email.strip().lower()
    password_hash, salt = _hash_password(password)
    try:
        with get_connection() as connection:
            cursor = connection.execute(
                """INSERT INTO users (email, password_hash, password_salt)
                   VALUES (?, ?, ?)""",
                (normalized_email, password_hash, salt),
            )
            user_id = cursor.lastrowid
            connection.execute(
                "INSERT INTO farmer_profiles (user_id, full_name) VALUES (?, ?)",
                (user_id, full_name.strip()),
            )
            connection.execute(
                "INSERT INTO activity_logs (user_id, event_type) VALUES (?, ?)",
                (user_id, "register"),
            )
        return True, "Account created. Please log in."
    except sqlite3.IntegrityError:
        return False, "An account with this email already exists."
    except sqlite3.Error:
        return False, "Could not create the account. Please try again."


def authenticate_user(email: str, password: str) -> dict | None:
    """Return safe user details on success; otherwise return None."""
    with get_connection() as connection:
        user = connection.execute(
            """SELECT id, email, password_hash, password_salt, role, is_active
               FROM users WHERE email = ? COLLATE NOCASE""",
            (email.strip().lower(),),
        ).fetchone()
        if not user or not user["is_active"]:
            return None
        candidate_hash, _ = _hash_password(password, user["password_salt"])
        if not hmac.compare_digest(candidate_hash, user["password_hash"]):
            return None
        connection.execute(
            "INSERT INTO activity_logs (user_id, event_type) VALUES (?, ?)",
            (user["id"], "login"),
        )
        return {"id": user["id"], "email": user["email"], "role": user["role"]}


def _show_auth_screen() -> None:
    """Render login, registration and guest access without altering app pages."""
    import streamlit as st
    st.markdown(
        """
        <div style="text-align:center;padding:1.2rem 0 0.4rem">
          <h1 style="color:#287a43;margin-bottom:0">AgriPulse</h1>
          <p style="opacity:.75">Where Agriculture Meets Intelligence</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    login_tab, register_tab = st.tabs(["Login", "Create account"])

    with login_tab:
        with st.form("agripulse_login_form"):
            email = st.text_input("Email address", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Login", use_container_width=True)
        if submitted:
            if not email.strip() or not password:
                st.error("Enter your email and password.")
            else:
                user = authenticate_user(email, password)
                if user:
                    st.session_state["agripulse_user"] = user
                    st.session_state["agripulse_access_mode"] = "user"
                    st.rerun()
                else:
                    st.error("Invalid email or password, or account is inactive.")

    with register_tab:
        st.caption("Demo storage notice: On Streamlit Community Cloud, accounts may disappear after an app restart or redeploy. Do not use a personal password here.")
        with st.form("agripulse_register_form"):
            full_name = st.text_input("Full name", key="register_name")
            new_email = st.text_input("Email address", key="register_email")
            new_password = st.text_input(
                "Create password (minimum 8 characters)",
                type="password",
                key="register_password",
            )
            confirm_password = st.text_input(
                "Confirm password", type="password", key="register_confirm"
            )
            submitted = st.form_submit_button(
                "Create farmer account", use_container_width=True
            )
        if submitted:
            error = validate_registration(
                full_name, new_email, new_password, confirm_password
            )
            if error:
                st.error(error)
            else:
                success, message = register_user(full_name, new_email, new_password)
                if success:
                    st.success(message)
                else:
                    st.error(message)

    st.divider()
    if st.button("Continue as guest", use_container_width=True):
        st.session_state["agripulse_access_mode"] = "guest"
        st.session_state["agripulse_user"] = None
        st.rerun()


def auth_gate() -> bool:
    """Return True when a signed-in farmer or guest may use the application."""
    import streamlit as st
    initialize_database()
    if st.session_state.get("agripulse_access_mode") == "guest":
        return True
    if st.session_state.get("agripulse_user"):
        return True
    _show_auth_screen()
    return False


def logout() -> None:
    """Clear the current AgriPulse authentication session."""
    import streamlit as st
    st.session_state.pop("agripulse_user", None)
    st.session_state.pop("agripulse_access_mode", None)
    st.rerun()
