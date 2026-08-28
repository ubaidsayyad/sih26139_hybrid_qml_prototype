import json
import pytest
from app import app, STATE, db
from core.models import User, Prediction
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

@pytest.fixture
def test_users(client):
    with app.app_context():
        user_a = User(name="Dr. Alice", email="alice.api@clinic.org", role="doctor")
        user_a.set_password("SecurePass123!")
        key_a = user_a.generate_api_key()

        user_b = User(name="Dr. Bob", email="bob.api@clinic.org", role="doctor")
        user_b.set_password("SecurePass123!")
        key_b = user_b.generate_api_key()

        db.session.add_all([user_a, user_b])
        db.session.commit()

        return {
            "user_a": {"id": user_a.id, "email": user_a.email, "key": key_a},
            "user_b": {"id": user_b.id, "email": user_b.email, "key": key_b}
        }

def test_api_health_endpoint(client):
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "healthy"
    assert "quantum_engine" in data
    assert "database" in data

def test_api_predict_authentication_missing_and_invalid_key(client, trained_pipeline):
    # 1. Missing X-API-Key header -> 401
    res_no_key = client.post("/api/v1/predict", json={"features": [0.1, 0.2, -0.1, 0.0]})
    assert res_no_key.status_code == 401
    data_no_key = res_no_key.get_json()
    assert data_no_key["status"] == "error"
    assert "Missing required 'X-API-Key'" in data_no_key["message"]

    # 2. Invalid X-API-Key header -> 401
    res_bad_key = client.post("/api/v1/predict",
                              headers={"X-API-Key": "qml_live_invalid_nonexistent_key"},
                              json={"features": [0.1, 0.2, -0.1, 0.0]})
    assert res_bad_key.status_code == 401
    data_bad_key = res_bad_key.get_json()
    assert "Invalid or revoked API key" in data_bad_key["message"]

def test_api_predict_input_validation(client, trained_pipeline, test_users):
    key = test_users["user_a"]["key"]
    headers = {"X-API-Key": key}

    # 1. No JSON payload -> 400
    res = client.post("/api/v1/predict", headers=headers, data="not a json")
    assert res.status_code == 400

    # 2. Missing 'features' field -> 400
    res = client.post("/api/v1/predict", headers=headers, json={"dataset": "Test"})
    assert res.status_code == 400
    assert "Missing required field 'features'" in res.get_json()["message"]

    # 3. Invalid dimension (e.g. 3 features instead of 4) -> 400
    res = client.post("/api/v1/predict", headers=headers, json={"features": [0.1, 0.2, 0.3]})
    assert res.status_code == 400
    assert "Expected exactly 4 PCA biomarker features" in res.get_json()["message"]

    # 4. Non-numeric features -> 400
    res = client.post("/api/v1/predict", headers=headers, json={"features": [0.1, "invalid", 0.3, 0.4]})
    assert res.status_code == 400

def test_api_predict_success_and_persistence(client, trained_pipeline, test_users):
    user_a = test_users["user_a"]
    headers = {"X-API-Key": user_a["key"]}
    payload = {
        "features": [0.35, -0.12, 0.08, -0.01],
        "dataset": "Breast Cancer Wisconsin (Diagnostic)"
    }

    res = client.post("/api/v1/predict", headers=headers, json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert data["prediction_id"] is not None
    assert "quantum" in data
    assert data["quantum"]["prediction"] in ("Malignant", "Benign")
    assert 0.0 <= data["quantum"]["confidence"] <= 1.0
    assert "classical" in data
    assert isinstance(data["consensus"], bool)

    # Verify prediction persisted in SQLite database tied to user_a
    with app.app_context():
        pred = db.session.get(Prediction, data["prediction_id"])
        assert pred is not None
        assert pred.user_id == user_a["id"]
        assert pred.dataset_name == "Breast Cancer Wisconsin (Diagnostic)"

def test_api_history_isolation(client, trained_pipeline, test_users):
    user_a = test_users["user_a"]
    user_b = test_users["user_b"]

    # User A makes 2 predictions via API
    for _ in range(2):
        client.post("/api/v1/predict", headers={"X-API-Key": user_a["key"]},
                    json={"features": [0.1, -0.2, 0.3, -0.1]})

    # User B makes 1 prediction via API
    client.post("/api/v1/predict", headers={"X-API-Key": user_b["key"]},
                json={"features": [-0.3, 0.1, 0.0, 0.2]})

    # User A requests /api/v1/history -> should have exactly 2 records
    res_a = client.get("/api/v1/history", headers={"X-API-Key": user_a["key"]})
    assert res_a.status_code == 200
    data_a = res_a.get_json()
    assert data_a["status"] == "success"
    assert data_a["total"] == 2
    assert len(data_a["predictions"]) == 2

    # User B requests /api/v1/history -> should have exactly 1 record
    res_b = client.get("/api/v1/history", headers={"X-API-Key": user_b["key"]})
    assert res_b.status_code == 200
    data_b = res_b.get_json()
    assert data_b["status"] == "success"
    assert data_b["total"] == 1
    assert len(data_b["predictions"]) == 1

def test_settings_and_key_regeneration(client, test_users):
    user_a = test_users["user_a"]
    old_key = user_a["key"]

    # 1. Login user A
    client.post("/login", data={"email": user_a["email"], "password": "SecurePass123!"})

    # 2. View settings
    res_settings = client.get("/settings")
    assert res_settings.status_code == 200
    assert old_key.encode() in res_settings.data

    # 3. Regenerate Key
    res_regen = client.post("/settings/regenerate-key", follow_redirects=True)
    assert res_regen.status_code == 200
    assert b"New API key issued successfully" in res_regen.data

    # 4. Confirm old key is now revoked and rejected by API
    res_old_key_denied = client.post("/api/v1/predict",
                                     headers={"X-API-Key": old_key},
                                     json={"features": [0.1, 0.2, 0.3, 0.4]})
    assert res_old_key_denied.status_code == 401

    # 5. Confirm new key works
    with app.app_context():
        u = db.session.get(User, user_a["id"])
        new_key = u.api_key
        assert new_key != old_key

    res_new_key_works = client.post("/api/v1/predict",
                                    headers={"X-API-Key": new_key},
                                    json={"features": [0.1, 0.2, 0.3, 0.4]})
    assert res_new_key_works.status_code == 200

def test_api_rate_limiting(client, trained_pipeline, test_users):
    from app import limiter
    limiter.enabled = True
    limiter.reset()

    user_a = test_users["user_a"]
    headers = {"X-API-Key": user_a["key"]}
    payload = {"features": [0.1, -0.1, 0.2, -0.2]}

    # Send 30 requests -> should all succeed (200)
    for _ in range(30):
        res = client.post("/api/v1/predict", headers=headers, json=payload)
        assert res.status_code == 200

    # 31st request -> should trigger 429 Too Many Requests
    res_limit = client.post("/api/v1/predict", headers=headers, json=payload)
    assert res_limit.status_code == 429
    assert res_limit.get_json()["error"] == "Rate Limit Exceeded"

    limiter.enabled = False
