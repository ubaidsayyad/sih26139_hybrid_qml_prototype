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
    role = db.Column(db.String(20), nullable=False, default="doctor")  # 'doctor', 'admin', or 'patient'
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

    @property
    def is_doctor(self) -> bool:
        return self.role == "doctor"

    @property
    def is_patient(self) -> bool:
        return self.role == "patient"

    @property
    def is_clinician(self) -> bool:
        return self.role in ("doctor", "admin")

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
    patient_record_id = db.Column(db.Integer, db.ForeignKey("patient_records.id", ondelete="SET NULL"), nullable=True, index=True)
    created_at = db.Column(db.DateTime, default=utc_now, nullable=False, index=True)

    # Relationships
    patient_record = db.relationship("PatientRecord", backref=db.backref("predictions", lazy="dynamic"))

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
            "patient_record_id": self.patient_record_id,
            "patient_name": self.patient_record.full_name if self.patient_record else None,
            "patient_age": self.patient_record.age if self.patient_record else None,
            "patient_gender": self.patient_record.gender if self.patient_record else None,
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


class PatientProfile(db.Model):
    __tablename__ = "patient_profiles"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False, index=True)
    full_name = db.Column(db.String(150), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(30), nullable=True)
    date_of_birth = db.Column(db.String(20), nullable=True)
    phone = db.Column(db.String(30), nullable=True)
    created_at = db.Column(db.DateTime, default=utc_now, nullable=False)

    # 1-to-1 Relationship with User
    user = db.relationship("User", backref=db.backref("patient_profile", uselist=False, cascade="all, delete-orphan"))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "full_name": self.full_name,
            "age": self.age,
            "gender": self.gender,
            "date_of_birth": self.date_of_birth,
            "phone": self.phone,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S") if self.created_at else None
        }

    def __repr__(self):
        return f"<PatientProfile #{self.id} User:{self.user_id} Name:'{self.full_name}' Age:{self.age}>"


class PatientRecord(db.Model):
    __tablename__ = "patient_records"

    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(150), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(30), nullable=True)
    entered_by = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    patient_user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    created_at = db.Column(db.DateTime, default=utc_now, nullable=False, index=True)

    # Relationships
    clinician = db.relationship("User", foreign_keys=[entered_by], backref=db.backref("entered_patient_records", lazy="dynamic"))
    patient_user = db.relationship("User", foreign_keys=[patient_user_id], backref=db.backref("patient_records", lazy="dynamic"))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "full_name": self.full_name,
            "age": self.age,
            "gender": self.gender,
            "entered_by": self.entered_by,
            "clinician_name": self.clinician.name if self.clinician else None,
            "patient_user_id": self.patient_user_id,
            "patient_email": self.patient_user.email if self.patient_user else None,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S") if self.created_at else None
        }

    def __repr__(self):
        return f"<PatientRecord #{self.id} '{self.full_name}' Age:{self.age} EnteredBy:{self.entered_by}>"

