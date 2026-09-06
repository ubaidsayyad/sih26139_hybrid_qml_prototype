import pytest
from app import app, db, STATE
from core.models import User, PatientProfile
from core.pipeline_runner import run_full_pipeline

@pytest.fixture
def client():
    app.config["TESTING"] = True
    app.config["WTF_CSRF_ENABLED"] = False
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    
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

def test_user_and_patient_profile_models(client):
    """Test User and PatientProfile model creation, relationships, and cascade delete."""
    with app.app_context():
        user = User(name="John Doe", email="john@example.com", role="patient")
        user.set_password("SecurePassword123")
        db.session.add(user)
        db.session.commit()

        assert user.is_patient is True
        assert user.is_doctor is False
        assert user.is_admin is False
        assert user.is_clinician is False

        profile = PatientProfile(
            user_id=user.id,
            full_name="John Jonathan Doe",
            age=42,
            gender="Male",
            date_of_birth="1984-05-12",
            phone="+1234567890"
        )
        db.session.add(profile)
        db.session.commit()

        assert user.patient_profile is not None
        assert user.patient_profile.age == 42
        assert user.patient_profile.gender == "Male"
        assert profile.user.email == "john@example.com"

        # Cascade delete test
        db.session.delete(user)
        db.session.commit()
        assert db.session.get(PatientProfile, profile.id) is None

def test_signup_doctor_vs_patient(client):
    """Test /signup creates proper roles and profiles."""
    # 1. Sign up as Doctor
    res_doc = client.post("/signup", data={
        "name": "Dr. Meredith Grey",
        "email": "meredith@seattlegrace.org",
        "role": "doctor",
        "password": "DoctorPassword123",
        "confirm_password": "DoctorPassword123"
    }, follow_redirects=False)

    assert res_doc.status_code == 302
    assert "/dashboard" in res_doc.headers["Location"]

    with app.app_context():
        doc = User.query.filter_by(email="meredith@seattlegrace.org").first()
        assert doc is not None
        assert doc.role == "doctor"
        assert doc.patient_profile is None

    client.get("/logout")

    # 2. Sign up as Patient without age -> should fail validation
    res_fail = client.post("/signup", data={
        "name": "Alice Patient",
        "email": "alice@patient.org",
        "role": "patient",
        "password": "PatientPassword123",
        "confirm_password": "PatientPassword123"
    }, follow_redirects=True)
    assert b"Please provide your age" in res_fail.data

    # 3. Sign up as Patient with valid profile fields
    res_pat = client.post("/signup", data={
        "name": "Alice Patient",
        "email": "alice@patient.org",
        "role": "patient",
        "patient_age": "34",
        "patient_gender": "Female",
        "patient_dob": "1992-03-15",
        "patient_phone": "555-0199",
        "password": "PatientPassword123",
        "confirm_password": "PatientPassword123"
    }, follow_redirects=False)

    assert res_pat.status_code == 302
    assert "/my-reports" in res_pat.headers["Location"]

    with app.app_context():
        pat = User.query.filter_by(email="alice@patient.org").first()
        assert pat is not None
        assert pat.role == "patient"
        assert pat.patient_profile is not None
        assert pat.patient_profile.age == 34
        assert pat.patient_profile.gender == "Female"
        assert pat.patient_profile.phone == "555-0199"

def test_admin_self_signup_prevented(client):
    """Confirm users cannot self-assign the admin role via /signup."""
    res = client.post("/signup", data={
        "name": "Hacker Admin",
        "email": "hacker@test.org",
        "role": "admin",
        "password": "HackerPassword123",
        "confirm_password": "HackerPassword123"
    }, follow_redirects=False)

    assert res.status_code == 302
    with app.app_context():
        u = User.query.filter_by(email="hacker@test.org").first()
        assert u is not None
        assert u.role == "doctor"  # fallback from disallowed admin

def test_login_role_based_redirects(client):
    """Test login redirect destinations based on user role."""
    with app.app_context():
        doc = User(name="Dr. House", email="house@princeton.org", role="doctor")
        doc.set_password("Vicodin123")
        pat = User(name="Bob Smith", email="bob@patients.org", role="patient")
        pat.set_password("BobPassword123")
        db.session.add_all([doc, pat])
        db.session.commit()

    # 1. Doctor login -> lands on /dashboard
    res_doc = client.post("/login", data={"email": "house@princeton.org", "password": "Vicodin123"}, follow_redirects=False)
    assert res_doc.status_code == 302
    assert "/dashboard" in res_doc.headers["Location"]
    client.get("/logout")

    # 2. Patient login -> lands on /my-reports
    res_pat = client.post("/login", data={"email": "bob@patients.org", "password": "BobPassword123"}, follow_redirects=False)
    assert res_pat.status_code == 302
    assert "/my-reports" in res_pat.headers["Location"]

    # 3. Patient login with clinical next URL -> redirected to /my-reports
    client.get("/logout")
    res_pat_next = client.post("/login?next=%2Fdashboard", data={"email": "bob@patients.org", "password": "BobPassword123"}, follow_redirects=False)
    assert res_pat_next.status_code == 302
    assert "/my-reports" in res_pat_next.headers["Location"]

def test_patient_route_protection(client, trained_pipeline):
    """Test that a logged in patient is denied access to all clinical/practitioner routes."""
    with app.app_context():
        pat = User(name="Charlie Patient", email="charlie@patients.org", role="role_temp")
        pat.role = "patient"
        pat.set_password("Charlie123")
        db.session.add(pat)
        db.session.commit()

    client.post("/login", data={"email": "charlie@patients.org", "password": "Charlie123"})

    # Check access to clinical routes
    protected_routes = [
        "/dashboard",
        "/predict",
        "/upload",
        "/history",
        "/settings",
        "/admin"
    ]

    for route in protected_routes:
        res = client.get(route, follow_redirects=False)
        assert res.status_code == 302, f"Route {route} did not redirect patient"
        assert "/my-reports" in res.headers["Location"], f"Route {route} redirected to {res.headers['Location']} instead of /my-reports"

    # Following redirect should display friendly restriction flash
    res_follow = client.get("/dashboard", follow_redirects=True)
    assert res_follow.status_code == 200
    assert b"Access restricted: Clinical and diagnostic tools are reserved for healthcare practitioners" in res_follow.data
    assert b"My Diagnostic Reports" in res_follow.data

def test_navbar_view_separation(client):
    """Test navbar HTML structure between Doctor and Patient sessions."""
    with app.app_context():
        doc = User(name="Dr. Stephen Strange", email="strange@hospital.org", role="doctor")
        doc.set_password("Agamotto123")
        pat = User(name="Peter Parker", email="peter@patients.org", role="patient")
        pat.set_password("WebSlinger123")
        db.session.add_all([doc, pat])
        db.session.commit()

    # 1. Doctor Navbar
    client.post("/login", data={"email": "strange@hospital.org", "password": "Agamotto123"})
    res_doc = client.get("/")
    doc_html = res_doc.get_data(as_text=True)
    assert "Dashboard" in doc_html
    assert "Try it" in doc_html
    assert "Upload Batch" in doc_html
    assert "History" in doc_html
    assert "Settings" in doc_html
    assert "Doctor" in doc_html
    assert "My Reports" not in doc_html
    assert "My Profile" not in doc_html
    client.get("/logout")

    # 2. Patient Navbar
    client.post("/login", data={"email": "peter@patients.org", "password": "WebSlinger123"})
    res_pat = client.get("/")
    pat_html = res_pat.get_data(as_text=True)
    assert "My Reports" in pat_html
    assert "My Profile" in pat_html
    assert "Patient" in pat_html
    assert "Sign Out" in pat_html
    # Clinical tools must be hidden
    assert "Dashboard" not in pat_html
    assert "Try it" not in pat_html
    assert "Upload Batch" not in pat_html
    assert "Admin Portal" not in pat_html
    assert "API Docs" not in pat_html
