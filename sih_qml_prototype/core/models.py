"""
models.py
---------
Database models and SQLAlchemy instance for the Hybrid QML Platform.
"""
from datetime import datetime, timezone
import json
import secrets
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()


def utc_now():
    return datetime.now(timezone.utc)


class User(db.Model, UserMixin):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(20), nullable=False, default="doctor")  # 'doctor' or 'admin'
    api_key = db.Column(db.String(64), unique=True, nullable=True, index=True)
    created_at = db.Column(db.DateTime, default=utc_now, nullable=False)

    # Relationships
    predictions = db.relationship(
        "Prediction",
        backref="user",
        lazy="dynamic",
        cascade="all, delete-orphan",
        order_by="desc(Prediction.created_at)"
    )

    def set_password(self, password: str):
        """Hashes and stores the user password."""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        """Verifies candidate password against stored hash."""
        return check_password_hash(self.password_hash, password)

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    def generate_api_key(self) -> str:
        """Generates and assigns a new cryptographically secure API key."""
        self.api_key = f"qml_live_{secrets.token_hex(24)}"
        return self.api_key

    def ensure_api_key(self) -> str:
        """Ensures the user has an active API key, generating one if missing."""
        if not self.api_key:
            return self.generate_api_key()
        return self.api_key

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "role": self.role,
            "api_key": self.api_key,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S") if self.created_at else None
        }

    def __repr__(self):
        return f"<User {self.email} ({self.role})>"


class Prediction(db.Model):
    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    dataset_name = db.Column(db.String(100), nullable=False, default="Breast Cancer Wisconsin (Diagnostic)")
    model_used = db.Column(db.String(50), nullable=False, default="both")  # 'quantum', 'classical', or 'both'
    input_features = db.Column(db.Text, nullable=False)  # JSON string of input features
    prediction_result = db.Column(db.String(50), nullable=False)  # 'Malignant' or 'Benign'
    confidence = db.Column(db.Float, nullable=False)  # Primary model confidence (0.0 to 1.0)
    true_label = db.Column(db.String(50), nullable=True)  # 'Malignant', 'Benign', or None
    metadata_json = db.Column(db.Text, nullable=True)  # Additional details (quantum_prob, classical_prob, etc)
    created_at = db.Column(db.DateTime, default=utc_now, nullable=False, index=True)

    @property
    def features_dict(self) -> dict:
        """Returns input features parsed from JSON."""
        try:
            return json.loads(self.input_features)
        except Exception:
            return {}

    @property
    def metadata_dict(self) -> dict:
        """Returns metadata parsed from JSON."""
        try:
            return json.loads(self.metadata_json) if self.metadata_json else {}
        except Exception:
            return {}

    @property
    def is_correct(self):
        if not self.true_label:
            return None
        return self.prediction_result.lower() == self.true_label.lower()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "user_name": self.user.name if self.user else None,
            "dataset_name": self.dataset_name,
            "model_used": self.model_used,
            "prediction_result": self.prediction_result,
            "confidence": round(self.confidence, 4),
            "true_label": self.true_label,
            "is_correct": self.is_correct,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S") if self.created_at else None,
            "features": self.features_dict,
            "metadata": self.metadata_dict
        }

    def __repr__(self):
        return f"<Prediction #{self.id} User:{self.user_id} Result:{self.prediction_result} ({self.confidence:.2f})>"
