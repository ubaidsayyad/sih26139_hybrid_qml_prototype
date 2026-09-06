import pytest
from app import app, db
from core.models import User

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

def test_find_care_page_unauthenticated_status(client):
    res = client.get("/find-care")
    assert res.status_code == 200
    html = res.data.decode("utf-8")
    
    # Assert Page Structure & Heading
    assert "Find Cancer Care" in html
    assert "Near You" in html
    
    # Assert Leaflet CDN & OpenStreetMap integration
    assert "leaflet@1.9.4/dist/leaflet.css" in html
    assert "leaflet@1.9.4/dist/leaflet.js" in html
    assert "tile.openstreetmap.org" in html
    
    # Assert Navigation Link
    assert 'href="/find-care"' in html or 'href="/find-care"' in html.replace(" ", "")
    assert "Find Care" in html

def test_find_care_privacy_disclaimer_present(client):
    res = client.get("/find-care")
    assert res.status_code == 200
    html = res.data.decode("utf-8")
    
    # Required clinical & privacy disclaimer
    assert "Location data is used only in your browser to search for nearby facilities" in html
    assert "never sent to or stored on our servers" in html
    assert "Facility data is community-sourced (OpenStreetMap)" in html
    assert "always verify details directly with the facility" in html

def test_find_care_ui_controls(client):
    res = client.get("/find-care")
    assert res.status_code == 200
    html = res.data.decode("utf-8")
    
    # Geolocation and manual fallback search controls
    assert "btn-use-location" in html
    assert "Use My Current Location" in html
    assert "input-city-search" in html
    assert "Enter city, town, or pincode / zip..." in html
    
    # Radius options
    assert 'data-radius="15000"' in html
    assert 'data-radius="30000"' in html
    assert 'data-radius="50000"' in html

    # States
    assert "find-care-loading" in html
    assert "find-care-empty" in html
    assert "find-care-error" in html
    assert "Get Directions" in html

def test_find_care_authenticated_doctor_and_patient(client):
    # Doctor access
    with app.app_context():
        doc = User(name="Dr. Oncology", email="oncology@clinic.org", role="doctor")
        doc.set_password("Password123!")
        pat = User(name="Patient Jane", email="jane@clinic.org", role="patient")
        pat.set_password("Password123!")
        db.session.add_all([doc, pat])
        db.session.commit()

    client.post("/login", data={"email": "oncology@clinic.org", "password": "Password123!"})
    res_doc = client.get("/find-care")
    assert res_doc.status_code == 200
    assert "Find Care" in res_doc.data.decode("utf-8")

    client.get("/logout")
    client.post("/login", data={"email": "jane@clinic.org", "password": "Password123!"})
    res_pat = client.get("/find-care")
    assert res_pat.status_code == 200
    assert "Find Care" in res_pat.data.decode("utf-8")
