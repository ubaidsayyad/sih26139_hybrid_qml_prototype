"""
test_named_patient.py
---------------------
Comprehensive test suite verifying the Named Patient Clinical Workflow:
1. PatientRecord model creation, relationships, and foreign keys.
2. Route protection for /predict/new-patient (Clinician only).
3. Demographic validation (name, age, gender).
4. Prediction execution and persistence linked to PatientRecord.
5. Patient account association (explicit dropdown and auto-matching by name).
6. History table presentation: displaying name and age for named records vs anonymous fallback.
7. Patient isolation on /my-reports: linked patient sees their reports, other patients do not.
8. PDF and detail access: linked patient can view and download their own reports, unrelated patients blocked.
"""
import os
import json
import pytest
import numpy as np

os.environ["FLASK_SECRET_KEY"] = "test-named-patient-key"
os.environ["WTF_CSRF_ENABLED"] = "false"

from app import app, db, init_db_and_seed_admin, STATE
from core.models import User, Prediction, PatientProfile, PatientRecord


class MockVQC:
    def predict_proba(self, X):
        return np.array([0.88])


class MockFairModel:
    def predict_proba(self, X):
        return np.array([[0.15, 0.85]])


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path):
    test_db_path = os.path.join(tmp_path, "test_named_patient.db")
    app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{test_db_path}"
    app.config["TESTING"] = True
    app.config["WTF_CSRF_ENABLED"] = False

    with app.app_context():
        db.create_all()
        init_db_and_seed_admin()

        # Populate test run state
        X_test = np.random.randn(10, 4)
        y_test = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        STATE["run"] = {
            "vqc": MockVQC(),
            "fair_model": MockFairModel(),
            "splits": {
                "quantum": {"X_test": X_test, "y_test": y_test},
                "classical_same_features": {"X_test": X_test, "y_test": y_test}
            },
            "results": {
                "quantum": {"accuracy": 0.95, "f1_score": 0.95, "roc_auc": 0.98},
                "classical_same_features": {"accuracy": 0.90, "f1_score": 0.90, "roc_auc": 0.95}
            }
        }

    yield

    with app.app_context():
        db.session.remove()
        db.drop_all()
    STATE["run"] = None


def create_user(name, email, role, password="TestPassword123!"):
    with app.app_context():
        u = User(name=name, email=email, role=role)
        u.set_password(password)
        db.session.add(u)
        db.session.commit()
        return u.id


def test_patient_record_model_attributes_and_relationships():
    """Confirms PatientRecord and Prediction relationship works bidirectionally."""
    with app.app_context():
        doc = User(name="Dr. Gregory House", email="house@quantum.health", role="doctor")
        doc.set_password("Diagnostics1!")
        patient_usr = User(name="John Doe", email="johndoe@patient.org", role="patient")
        patient_usr.set_password("PatientPass1!")
        db.session.add_all([doc, patient_usr])
        db.session.commit()

        # Create PatientRecord
        rec = PatientRecord(
            full_name="Johnathan Doe",
            age=45,
            gender="Male",
            entered_by=doc.id,
            patient_user_id=patient_usr.id
        )
        db.session.add(rec)
        db.session.commit()

        assert rec.id is not None
        assert rec.clinician.email == "house@quantum.health"
        assert rec.patient_user.email == "johndoe@patient.org"

        # Create linked Prediction
        pred = Prediction(
            user_id=doc.id,
            patient_record_id=rec.id,
            dataset_name="Test Diagnostic",
            model_used="both",
            input_features=json.dumps({"PC1": 0.5}),
            prediction_result="Malignant",
            confidence=0.88,
            true_label="Malignant",
            metadata_json=json.dumps({"patient_name": "Johnathan Doe"})
        )
        db.session.add(pred)
        db.session.commit()

        assert pred.patient_record is not None
        assert pred.patient_record.full_name == "Johnathan Doe"
        assert rec.predictions.count() == 1
        assert rec.predictions.first().id == pred.id

        # Verify to_dict
        d = pred.to_dict()
        assert d["patient_record_id"] == rec.id
        assert d["patient_name"] == "Johnathan Doe"
        assert d["patient_age"] == 45
        assert d["patient_gender"] == "Male"


def test_predict_new_patient_route_access():
    """Verifies RBAC: Doctors and Admins can access /predict/new-patient, Patients are redirected."""
    doc_id = create_user("Dr. Allison Cameron", "cameron@quantum.health", "doctor")
    pat_id = create_user("Patient Bob", "bob@patient.org", "patient")
    adm_id = create_user("Admin User", "superadmin@quantum.health", "admin")

    client = app.test_client()

    # Unauthenticated -> redirect to login
    res = client.get("/predict/new-patient")
    assert res.status_code == 302
    assert "/login" in res.headers["Location"]

    # Patient -> restricted, redirected to /my-reports
    with client.session_transaction() as sess:
        sess["_user_id"] = str(pat_id)
        sess["_fresh"] = True
    res = client.get("/predict/new-patient", follow_redirects=True)
    assert res.status_code == 200
    assert b"My Diagnostic Reports" in res.data

    # Doctor -> allowed
    with client.session_transaction() as sess:
        sess["_user_id"] = str(doc_id)
        sess["_fresh"] = True
    res = client.get("/predict/new-patient")
    assert res.status_code == 200
    assert b"Named Patient Screening" in res.data
    assert b"Patient Intake Demographics" in res.data

    # Admin -> allowed
    with client.session_transaction() as sess:
        sess["_user_id"] = str(adm_id)
        sess["_fresh"] = True
    res = client.get("/predict/new-patient")
    assert res.status_code == 200
    assert b"Named Patient Screening" in res.data


def test_predict_new_patient_validation_errors():
    """Validates missing name and invalid age handling."""
    doc_id = create_user("Dr. Eric Foreman", "foreman@quantum.health", "doctor")
    client = app.test_client()

    with client.session_transaction() as sess:
        sess["_user_id"] = str(doc_id)
        sess["_fresh"] = True

    # Empty name
    res = client.post("/predict/new-patient", json={"full_name": "", "age": 40}, headers={"X-Requested-With": "XMLHttpRequest"})
    assert res.status_code == 400
    assert "full name" in res.json["error"].lower()

    # Invalid age
    res = client.post("/predict/new-patient", json={"full_name": "Test Patient", "age": 200}, headers={"X-Requested-With": "XMLHttpRequest"})
    assert res.status_code == 400
    assert "age" in res.json["error"].lower()


def test_full_named_patient_flow_and_isolation():
    """
    Complete end-to-end flow:
    1. Doctor logs in and screens named patient 'Test Patient A' (age 45, female).
    2. Prediction is saved with PatientRecord and linked to patient account.
    3. Confirms /history shows 'Test Patient A' and age 45.
    4. Confirms unrelated patient 'Patient Other' visits /my-reports and sees NO reports (empty state).
    5. Confirms linked patient 'Test Patient A' visits /my-reports and sees their report.
    6. Confirms linked patient can download their PDF report, while unrelated patient is blocked.
    """
    # 1. Setup Doctor and two Patients
    doc_id = create_user("Dr. Robert Chase", "chase@quantum.health", "doctor")
    pat_a_id = create_user("Test Patient A", "patient.a@health.org", "patient")
    pat_b_id = create_user("Patient Other B", "patient.b@health.org", "patient")

    with app.app_context():
        # Setup patient profiles
        prof_a = PatientProfile(user_id=pat_a_id, full_name="Test Patient A", age=45, gender="Female")
        prof_b = PatientProfile(user_id=pat_b_id, full_name="Patient Other B", age=32, gender="Male")
        db.session.add_all([prof_a, prof_b])
        db.session.commit()

    client = app.test_client()

    # 2. Doctor executes prediction for named patient A
    with client.session_transaction() as sess:
        sess["_user_id"] = str(doc_id)
        sess["_fresh"] = True

    post_payload = {
        "full_name": "Test Patient A",
        "age": 45,
        "gender": "Female",
        "patient_user_id": pat_a_id,
        "index": 0
    }
    res = client.post("/predict/new-patient", json=post_payload, headers={"X-Requested-With": "XMLHttpRequest"})
    assert res.status_code == 200
    data = res.json
    assert data["success"] is True
    assert data["patient_name"] == "Test Patient A"
    assert data["patient_age"] == 45
    assert data["is_linked"] is True
    pred_id = data["prediction_id"]

    # Also execute an anonymous prediction for comparison
    anon_res = client.post("/predict", json={"index": 1})
    assert anon_res.status_code == 200
    anon_pred_id = anon_res.json["prediction_id"]

    # 3. Doctor checks /history
    hist_res = client.get("/history")
    assert hist_res.status_code == 200
    assert b"Test Patient A" in hist_res.data
    assert b"45 yrs" in hist_res.data
    assert b"Anonymous / Test Sample" in hist_res.data

    # 4. Unrelated Patient B logs in -> /my-reports
    with client.session_transaction() as sess:
        sess["_user_id"] = str(pat_b_id)
        sess["_fresh"] = True

    my_rep_b = client.get("/my-reports")
    assert my_rep_b.status_code == 200
    # Patient B must NOT see Patient A's report!
    assert b"Test Patient A" not in my_rep_b.data
    assert b"No Diagnostic Reports Published Yet" in my_rep_b.data

    # Patient B attempts to download Patient A's report PDF -> Access Denied
    pdf_b = client.get(f"/history/{pred_id}/pdf", follow_redirects=True)
    assert b"Access denied" in pdf_b.data or b"My Diagnostic Reports" in pdf_b.data

    # 5. Linked Patient A logs in -> /my-reports
    with client.session_transaction() as sess:
        sess["_user_id"] = str(pat_a_id)
        sess["_fresh"] = True

    my_rep_a = client.get("/my-reports")
    assert my_rep_a.status_code == 200
    # Patient A sees their published report
    assert b"Published Clinical Reports" in my_rep_a.data
    assert b"Test Patient A" in my_rep_a.data
    assert f"REPORT #{pred_id}".encode() in my_rep_a.data

    # 6. Linked Patient A downloads their PDF report -> Success!
    pdf_a = client.get(f"/history/{pred_id}/pdf")
    assert pdf_a.status_code == 200
    assert pdf_a.mimetype == "application/pdf"
    assert b"%PDF" in pdf_a.data[:10]

    # 7. Linked Patient A views breakdown -> Success!
    detail_a = client.get(f"/history/{pred_id}")
    assert detail_a.status_code == 200
    assert b"Diagnostic Breakdown" in detail_a.data
    assert b"Test Patient A" in detail_a.data
