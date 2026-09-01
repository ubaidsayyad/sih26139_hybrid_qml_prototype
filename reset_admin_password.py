#!/usr/bin/env python3
"""
reset_admin_password.py
-----------------------
Standalone utility to reset the password for the admin@quantum.health account.
Uses the identical password hashing method (werkzeug.security.generate_password_hash)
as the Flask application, ensuring authentication works properly.

Usage:
    python reset_admin_password.py <new_password> [email]
"""
import sys
import os

# Ensure package directory is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(SCRIPT_DIR) == "sih_qml_prototype":
    PKG_DIR = SCRIPT_DIR
else:
    PKG_DIR = os.path.join(SCRIPT_DIR, "sih_qml_prototype")

if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)

try:
    from app import app, db
    from core.models import User
except ImportError as e:
    print(f"[ERROR] Failed to import app/models from {PKG_DIR}: {e}")
    sys.exit(1)


def reset_admin_password(new_password: str, email: str = "admin@quantum.health") -> bool:
    """Updates the admin account password in instance/app.db using app's hashing method."""
    if not new_password or len(new_password.strip()) == 0:
        print("[ERROR] Password cannot be empty.")
        return False

    with app.app_context():
        # Ensure database tables exist
        db.create_all()

        admin = User.query.filter_by(email=email).first()
        if not admin:
            # Fallback: check if an admin with another email exists
            admin = User.query.filter_by(role="admin").first()

        if not admin:
            print(f"[INFO] No existing admin account found. Creating new admin account for {email}...")
            admin = User(
                name="System Administrator",
                email=email,
                role="admin"
            )
            admin.generate_api_key()
            db.session.add(admin)
        else:
            admin.email = email

        # Hash and set new password using User.set_password
        admin.set_password(new_password)
        db.session.commit()

        # Immediate verification
        refetched_admin = User.query.filter_by(email=email).first()
        if refetched_admin and refetched_admin.check_password(new_password):
            print("=" * 65)
            print("[SUCCESS] Admin password successfully updated in database!")
            print(f"  Account Email  : {refetched_admin.email}")
            print(f"  Account Role   : {refetched_admin.role}")
            print(f"  API Key        : {refetched_admin.api_key}")
            print(f"  Password Check : VERIFIED (check_password returned True)")
            print("=" * 65)
            return True
        else:
            print("[ERROR] Password verification failed after commit.")
            return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python reset_admin_password.py <new_password> [email]")
        print("Example: python reset_admin_password.py AdminSecure2026!")
        sys.exit(1)

    password_arg = sys.argv[1]
    email_arg = sys.argv[2] if len(sys.argv) > 2 else "admin@quantum.health"

    success = reset_admin_password(password_arg, email_arg)
    if not success:
        sys.exit(1)
