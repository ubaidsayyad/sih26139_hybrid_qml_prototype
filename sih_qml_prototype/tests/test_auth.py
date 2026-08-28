import pytest
from app import app
from core.models import db, User

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

def test_user_password_hashing():
    u = User(name="Dr. Alice", email="alice@hospital.org", role="doctor")
    u.set_password("SuperSecretPassword123")
    assert u.password_hash != "SuperSecretPassword123"
    assert u.check_password("SuperSecretPassword123") is True
    assert u.check_password("WrongPassword") is False

def test_unauthenticated_protected_routes(client):
    protected_routes = ["/dashboard", "/predict", "/upload", "/admin"]
    for route in protected_routes:
        res = client.get(route)
        assert res.status_code == 302
        assert "/login" in res.headers["Location"]

def test_signup_validation_short_password(client):
    res = client.post("/signup", data={
        "name": "Dr. Short",
        "email": "short@test.org",
        "role": "doctor",
        "password": "short",
        "confirm_password": "short"
    }, follow_redirects=True)
    assert b"at least 8 characters" in res.data

def test_signup_validation_mismatched_password(client):
    res = client.post("/signup", data={
        "name": "Dr. Mismatch",
        "email": "mismatch@test.org",
        "role": "doctor",
        "password": "Password123",
        "confirm_password": "Password456"
    }, follow_redirects=True)
    assert b"Passwords do not match" in res.data

def test_signup_validation_invalid_email(client):
    res = client.post("/signup", data={
        "name": "Dr. BadEmail",
        "email": "not-an-email",
        "role": "doctor",
        "password": "Password123",
        "confirm_password": "Password123"
    }, follow_redirects=True)
    assert b"valid corporate or clinical email" in res.data

def test_signup_success_and_login(client):
    res = client.post("/signup", data={
        "name": "Dr. Sarah Connor",
        "email": "sarah@cyber.health",
        "role": "doctor",
        "password": "QuantumSecurePass123",
        "confirm_password": "QuantumSecurePass123"
    }, follow_redirects=True)
    assert res.status_code == 200
    assert b"Account successfully registered" in res.data

    with app.app_context():
        user = User.query.filter_by(email="sarah@cyber.health").first()
        assert user is not None
        assert user.name == "Dr. Sarah Connor"
        assert user.role == "doctor"

def test_signup_duplicate_email(client):
    client.post("/signup", data={
        "name": "User One",
        "email": "dup@test.org",
        "role": "doctor",
        "password": "Password123",
        "confirm_password": "Password123"
    })

    # Log out so the next request is from an unauthenticated user attempting duplicate registration
    client.get("/logout")

    # Attempt duplicate registration
    res = client.post("/signup", data={
        "name": "User Two",
        "email": "dup@test.org",
        "role": "doctor",
        "password": "Password123",
        "confirm_password": "Password123"
    }, follow_redirects=True)
    assert b"already registered" in res.data

def test_login_invalid_credentials(client):
    res = client.post("/login", data={
        "email": "nonexistent@hospital.org",
        "password": "WrongPassword"
    }, follow_redirects=True)
    assert b"Invalid email address or password" in res.data

def test_login_success_and_redirect_next(client):
    # Create doctor account
    with app.app_context():
        doc = User(name="Dr. Gregory House", email="house@princeton.edu", role="doctor")
        doc.set_password("Vicodin123")
        db.session.add(doc)
        db.session.commit()

    # Login with next parameter pointing to /predict
    res = client.post("/login?next=%2Fpredict", data={
        "email": "house@princeton.edu",
        "password": "Vicodin123"
    }, follow_redirects=True)
    assert res.status_code == 200
    assert b"Welcome back, Dr. Gregory House!" in res.data

def test_role_based_access_doctor_vs_admin(client):
    with app.app_context():
        doc = User(name="Dr. Doctor", email="doc@test.org", role="doctor")
        doc.set_password("DoctorPass123")
        admin = User(name="Admin User", email="admin@test.org", role="admin")
        admin.set_password("AdminPass123")
        db.session.add_all([doc, admin])
        db.session.commit()

    # 1. Login as doctor -> try accessing /admin
    client.post("/login", data={"email": "doc@test.org", "password": "DoctorPass123"})
    res = client.get("/admin", follow_redirects=True)
    assert b"Access denied: System Administrator privileges required" in res.data

    # 2. Logout doctor and login as admin -> access /admin
    client.get("/logout")
    client.post("/login", data={"email": "admin@test.org", "password": "AdminPass123"})
    res_admin = client.get("/admin")
    assert res_admin.status_code == 200
    assert b"Administrator Console" in res_admin.data
    assert b"Registered System Accounts" in res_admin.data

def test_logout_session_termination(client):
    # Register & login
    client.post("/signup", data={
        "name": "Dr. Logout Test",
        "email": "logout@test.org",
        "role": "doctor",
        "password": "Password123",
        "confirm_password": "Password123"
    })

    # Logout
    res_logout = client.get("/logout", follow_redirects=True)
    assert b"signed out safely" in res_logout.data or b"signed out" in res_logout.data

    # Access protected route should now redirect
    res_protected = client.get("/dashboard")
    assert res_protected.status_code == 302
    assert "/login" in res_protected.headers["Location"]
