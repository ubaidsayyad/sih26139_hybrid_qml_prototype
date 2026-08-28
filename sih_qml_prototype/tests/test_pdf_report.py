import json
import pytest
from datetime import datetime
from app import app, STATE
from core.models import db, User, Prediction
from core.pdf_report import generate_prediction_pdf
from core.pipeline_runner import run_full_pipeline

@pytest.fixture
def client():
    app.config["TESTING"] = True
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    app.config["WTF_CSRF_ENABLED"] = False
    app.config["RATELIMIT_ENABLED"] = False

    from app import limiter
    limiter.enabled = False

    with app.test_client() as client:
        with app.app_context():
            db.create_all()
            yield client
            db.session.remove()
            db.drop_all()

@pytest.fixture
def trained_pipeline():
    if STATE.get("run") is None:
        STATE["run"] = run_full_pipeline(epochs=2, quantum_backend="simulator")
    return STATE["run"]

class MockPrediction:
    def __init__(self, id, res, conf, true_label):
        self.id = id
        self.prediction_result = res
        self.confidence = conf
        self.true_label = true_label
        self.dataset_name = "Breast Cancer Wisconsin (Diagnostic)"
        self.created_at = datetime(2026, 8, 28, 14, 30, 0)
        self.features_dict = {"PC1": 0.25, "PC2": -0.15, "PC3": 0.08, "PC4": -0.02}
        self.metadata_dict = {
            "quantum_prediction": res,
            "quantum_prob": conf,
            "quantum_conf": conf,
            "classical_prediction": res,
            "classical_prob": conf,
            "classical_conf": conf
        }

def test_generate_pdf_structure_and_header():
    pred_mal = MockPrediction(101, "Malignant", 0.885, "Malignant")
    pdf_bytes = generate_prediction_pdf(pred_mal, "Dr. Alice", {
        "quantum_explainability": {"importances": {"PC1": 0.15, "PC2": 0.08}}
    })
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF-")

    pred_ben = MockPrediction(102, "Benign", 0.942, "Benign")
    pdf_bytes_ben = generate_prediction_pdf(pred_ben, "Dr. Bob")
    assert isinstance(pdf_bytes_ben, bytes)
    assert len(pdf_bytes_ben) > 1000
    assert pdf_bytes_ben.startswith(b"%PDF-")

def test_unauthenticated_pdf_download_redirect(client):
    res = client.get("/history/1/pdf")
    assert res.status_code == 302
    assert "/login" in res.headers["Location"]

def test_authenticated_doctor_pdf_download(client, trained_pipeline):
    # 1. Signup doctor
    client.post("/signup", data={
        "name": "Dr. Sarah Connor",
        "email": "sarah.pdf@quantum.health",
        "role": "doctor",
        "password": "Password123!",
        "confirm_password": "Password123!"
    })

    # 2. Run prediction
    res_pred = client.post("/predict", json={"index": 1})
    assert res_pred.status_code == 200
    pred_id = res_pred.get_json()["prediction_id"]
    assert pred_id is not None

    # 3. Download PDF report
    res_pdf = client.get(f"/history/{pred_id}/pdf")
    assert res_pdf.status_code == 200
    assert res_pdf.headers["Content-Type"] == "application/pdf"
    assert "attachment; filename=report_patient_" in res_pdf.headers["Content-Disposition"]
    assert res_pdf.data.startswith(b"%PDF-")
    assert len(res_pdf.data) > 1000

def test_doctor_user_isolation_on_pdf_download(client):
    with app.app_context():
        doc_a = User(name="Dr. Alice", email="alice.pdf@clinic.org", role="doctor")
        doc_a.set_password("Password123!")
        doc_b = User(name="Dr. Bob", email="bob.pdf@clinic.org", role="doctor")
        doc_b.set_password("Password123!")
        db.session.add_all([doc_a, doc_b])
        db.session.commit()

        # Alice creates a prediction
        pred_alice = Prediction(
            user_id=doc_a.id,
            dataset_name="Breast Cancer Wisconsin",
            model_used="both",
            input_features=json.dumps({"PC1": 0.5}),
            prediction_result="Malignant",
            confidence=0.92,
            true_label="Malignant"
        )
        db.session.add(pred_alice)
        db.session.commit()
        alice_pred_id = pred_alice.id

    # 1. Bob logs in and tries to download Alice's PDF -> Denied
    client.post("/login", data={"email": "bob.pdf@clinic.org", "password": "Password123!"})
    res_bob_denied = client.get(f"/history/{alice_pred_id}/pdf", follow_redirects=True)
    assert b"Access denied: You cannot export other practitioners" in res_bob_denied.data

    # 2. Alice logs in and downloads her own PDF -> Success
    client.get("/logout")
    client.post("/login", data={"email": "alice.pdf@clinic.org", "password": "Password123!"})
    res_alice_pdf = client.get(f"/history/{alice_pred_id}/pdf")
    assert res_alice_pdf.status_code == 200
    assert res_alice_pdf.headers["Content-Type"] == "application/pdf"

def test_admin_can_download_any_doctor_pdf(client):
    with app.app_context():
        doc = User(name="Dr. Resident", email="resident@clinic.org", role="doctor")
        doc.set_password("Password123!")
        admin = User(name="Chief Admin", email="admin.pdf@clinic.org", role="admin")
        admin.set_password("Password123!")
        db.session.add_all([doc, admin])
        db.session.commit()

        pred = Prediction(
            user_id=doc.id,
            dataset_name="Breast Cancer Wisconsin",
            model_used="both",
            input_features=json.dumps({"PC1": 0.1}),
            prediction_result="Benign",
            confidence=0.97
        )
        db.session.add(pred)
        db.session.commit()
        pred_id = pred.id

    # Admin logs in -> can download doctor's PDF
    client.post("/login", data={"email": "admin.pdf@clinic.org", "password": "Password123!"})
    res_admin_pdf = client.get(f"/history/{pred_id}/pdf")
    assert res_admin_pdf.status_code == 200
    assert res_admin_pdf.headers["Content-Type"] == "application/pdf"
    assert res_admin_pdf.data.startswith(b"%PDF-")
