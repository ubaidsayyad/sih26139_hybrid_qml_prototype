import pytest
from app import app

@pytest.fixture
def client():
    app.config["TESTING"] = True
    app.config["WTF_CSRF_ENABLED"] = False
    with app.test_client() as client:
        yield client

def test_about_page_status_and_content(client):
    """Confirm /about route loads with HTTP 200 and contains all Team Platinum members."""
    response = client.get("/about")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    
    # Assert section title and heading
    assert "TEAM PLATINUM" in html
    assert "The team behind this platform." in html
    assert "Kavikulguru Institute of Technology and Science" in html
    
    # Assert all 6 members
    assert "Pranjal Madkam" in html
    assert "Ubaid Sayyad" in html
    assert "Saloni Bopche" in html
    assert "Sheshandri Lajurkar" in html
    assert "Roshani Tijare" in html
    assert "Anshu Ambule" in html
    
    # Assert roll numbers
    assert "IT25015" in html
    assert "IT25064" in html
    assert "IT25019" in html
    assert "CS25038" in html
    assert "CS25043" in html
    assert "CT25058" in html

def test_overview_navbar_links_to_about(client):
    """Confirm navigation bar has About Us link and Overview page doesn't contain team section."""
    response = client.get("/")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    
    assert 'href="/about"' in html
    assert "About Us" in html
    # Assert team cards are not duplicated on index
    assert "IT25015" not in html
