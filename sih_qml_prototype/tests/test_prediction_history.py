import json
import pytest
from app import app, STATE
from core.models import db, User, Prediction
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
    """Initializes a lightweight trained pipeline in STATE for testing inferences."""
    if STATE.get("run") is None:
        run_data = run_full_pipeline(epochs=2, quantum_backend="simulator")
        STATE["run"] = run_data
    return STATE["run"]

def test_prediction_model_properties(client):
    with app.app_context():
        u = User(name="Dr. Tester", email="tester@quantum.health", role="doctor")
        u.set_password("SecurePass123!")
        db.session.add(u)
        db.session.commit()

        feats = {"PC1": 0.45, "PC2": -0.12}
        meta = {"quantum_prob": 0.88, "quantum_prediction": "Malignant"}
        p = Prediction(
            user_id=u.id,
            dataset_name="Breast Cancer Wisconsin",
            model_used="both",
            input_features=json.dumps(feats),
            prediction_result="Malignant",
            confidence=0.88,
            true_label="Malignant",
            metadata_json=json.dumps(meta)
        )
        db.session.add(p)
        db.session.commit()

        assert p.features_dict == feats
        assert p.metadata_dict == meta
        assert p.is_correct is True
        d = p.to_dict()
        assert d["prediction_result"] == "Malignant"
        assert d["confidence"] == 0.88
        assert d["user_name"] == "Dr. Tester"

def test_unauthenticated_history_redirect(client):
    res = client.get("/history")
    assert res.status_code == 302
    assert "/login" in res.headers["Location"]

    res_detail = client.get("/history/1")
    assert res_detail.status_code == 302
    assert "/login" in res_detail.headers["Location"]

def test_history_empty_for_new_doctor(client):
    client.post("/signup", data={
        "name": "Dr. Fresh",
        "email": "fresh@quantum.health",
        "role": "doctor",
        "password": "Password123!",
        "confirm_password": "Password123!"
    })
    res = client.get("/history")
    assert res.status_code == 200
    assert b"No diagnostic records found" in res.data

def test_predict_auto_records_to_db(client, trained_pipeline):
    # Signup and login
    client.post("/signup", data={
        "name": "Dr. Predictor",
        "email": "predictor@quantum.health",
        "role": "doctor",
        "password": "Password123!",
        "confirm_password": "Password123!"
    })

    # Execute single prediction
    res = client.post("/predict", json={"index": 2})
    assert res.status_code == 200

    # Verify prediction row created in database
    with app.app_context():
        user = User.query.filter_by(email="predictor@quantum.health").first()
        preds = Prediction.query.filter_by(user_id=user.id).all()
        assert len(preds) == 1
        assert preds[0].dataset_name == "Breast Cancer Wisconsin (Diagnostic)"
        assert preds[0].model_used == "both"
        assert preds[0].confidence > 0.0

    # Verify record appears in /history HTML table
    res_hist = client.get("/history")
    assert res_hist.status_code == 200
    assert b"#1" in res_hist.data or b"Breast Cancer Wisconsin" in res_hist.data

def test_doctor_user_isolation(client, trained_pipeline):
    with app.app_context():
        doc_a = User(name="Dr. Alice", email="alice@clinic.org", role="doctor")
        doc_a.set_password("Password123!")
        doc_b = User(name="Dr. Bob", email="bob@clinic.org", role="doctor")
        doc_b.set_password("Password123!")
        db.session.add_all([doc_a, doc_b])
        db.session.commit()

        # Alice creates a prediction
        pred_alice = Prediction(
            user_id=doc_a.id,
            dataset_name="Breast Cancer Wisconsin",
            model_used="quantum",
            input_features=json.dumps({"PC1": 0.5}),
            prediction_result="Malignant",
            confidence=0.92,
            true_label="Malignant"
        )
        db.session.add(pred_alice)
        db.session.commit()
        alice_pred_id = pred_alice.id

    # 1. Login as Bob -> visit /history -> should NOT see Alice's record
    client.post("/login", data={"email": "bob@clinic.org", "password": "Password123!"})
    res_bob_hist = client.get("/history")
    assert b"No diagnostic records found" in res_bob_hist.data

    # 2. Bob tries to access Alice's prediction detail directly
    res_bob_detail = client.get(f"/history/{alice_pred_id}", follow_redirects=True)
    assert b"Access denied: You cannot view other practitioners" in res_bob_detail.data

    # 3. Login as Alice -> Alice CAN access her own prediction detail
    client.get("/logout")
    client.post("/login", data={"email": "alice@clinic.org", "password": "Password123!"})
    res_alice_detail = client.get(f"/history/{alice_pred_id}")
    assert res_alice_detail.status_code == 200
    assert b"Diagnostic Breakdown" in res_alice_detail.data

def test_history_pagination_and_sorting(client):
    with app.app_context():
        doc = User(name="Dr. Multi", email="multi@clinic.org", role="doctor")
        doc.set_password("Password123!")
        db.session.add(doc)
        db.session.commit()

        # Create 25 predictions
        for i in range(25):
            p = Prediction(
                user_id=doc.id,
                dataset_name="Breast Cancer Wisconsin",
                model_used="both" if i % 2 == 0 else "quantum",
                input_features=json.dumps({"PC1": i * 0.01}),
                prediction_result="Malignant" if i % 3 == 0 else "Benign",
                confidence=round(0.5 + (i * 0.01), 2),
                true_label="Malignant" if i % 3 == 0 else "Benign"
            )
            db.session.add(p)
        db.session.commit()

    client.post("/login", data={"email": "multi@clinic.org", "password": "Password123!"})

    # Page 1 (20 items)
    res_p1 = client.get("/history?page=1")
    assert res_p1.status_code == 200
    assert b"Page 1 of 2" in res_p1.data

    # Page 2 (5 items)
    res_p2 = client.get("/history?page=2")
    assert res_p2.status_code == 200
    assert b"Page 2 of 2" in res_p2.data

    # Filter by result=Malignant
    res_filt = client.get("/history?result=Malignant")
    assert res_filt.status_code == 200

def test_admin_all_history_access(client):
    with app.app_context():
        doc1 = User(name="Dr. Alpha", email="alpha@med.org", role="doctor")
        doc1.set_password("Password123!")
        doc2 = User(name="Dr. Beta", email="beta@med.org", role="doctor")
        doc2.set_password("Password123!")
        admin = User(name="Admin Boss", email="boss@med.org", role="admin")
        admin.set_password("Password123!")
        db.session.add_all([doc1, doc2, admin])
        db.session.commit()

        # Insert predictions for both doctors
        p1 = Prediction(user_id=doc1.id, dataset_name="TestDS", model_used="both", input_features="{}", prediction_result="Malignant", confidence=0.91)
        p2 = Prediction(user_id=doc2.id, dataset_name="TestDS", model_used="quantum", input_features="{}", prediction_result="Benign", confidence=0.84)
        db.session.add_all([p1, p2])
        db.session.commit()

    # 1. Doctor tries accessing /admin/all-history -> Denied
    client.post("/login", data={"email": "alpha@med.org", "password": "Password123!"})
    res_doc = client.get("/admin/all-history", follow_redirects=True)
    assert b"Access denied: System Administrator privileges required" in res_doc.data

    # 2. Admin logs in -> accesses /admin/all-history -> sees both doctors' records
    client.get("/logout")
    client.post("/login", data={"email": "boss@med.org", "password": "Password123!"})
    res_admin = client.get("/admin/all-history")
    assert res_admin.status_code == 200
    assert b"Global Prediction Audit Trail" in res_admin.data
    assert b"Dr. Alpha" in res_admin.data
    assert b"Dr. Beta" in res_admin.data
