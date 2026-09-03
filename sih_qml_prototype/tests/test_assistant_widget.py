import os
import subprocess
import json
import pytest
from app import app, db, STATE
from core.models import User
from core.pipeline_runner import run_full_pipeline

@pytest.fixture
def auth_client():
    app.config["TESTING"] = True
    app.config["WTF_CSRF_ENABLED"] = False
    
    if STATE.get("run") is None:
        STATE["run"] = run_full_pipeline(epochs=2, quantum_backend="simulator")

    with app.test_client() as client:
        with app.app_context():
            db.create_all()
            user = User.query.filter_by(email="doctor_test_assistant@quantum.health").first()
            if not user:
                user = User(
                    name="Dr. Assistant Tester",
                    email="doctor_test_assistant@quantum.health",
                    role="doctor"
                )
                user.set_password("SecureDocPass123!")
                user.generate_api_key()
                db.session.add(user)
                db.session.commit()
            
            # Log in
            client.post("/login", data={
                "email": "doctor_test_assistant@quantum.health",
                "password": "SecureDocPass123!"
            })
        yield client

def test_assistant_present_on_all_pages(auth_client):
    """Verify assistant widget is injected on landing, about, predict, upload, history, settings, dashboard."""
    pages = ["/", "/about", "/predict", "/upload", "/history", "/settings", "/dashboard"]
    
    for page in pages:
        res = auth_client.get(page)
        assert res.status_code == 200, f"Page {page} failed to load with status {res.status_code}"
        html = res.get_data(as_text=True)
        assert 'id="platform-assistant-widget"' in html, f"Missing assistant widget on {page}"
        assert 'id="assistant-toggle-btn"' in html, f"Missing toggle button on {page}"
        assert 'id="assistant-panel"' in html, f"Missing assistant panel on {page}"
        assert 'assistant.js' in html, f"Missing assistant.js script inclusion on {page}"

def test_assistant_js_rule_matching_with_node():
    """Execute assistant.js intent matching directly in Node.js across all required question variations."""
    static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
    js_path = os.path.join(static_dir, "assistant.js").replace("\\", "/")
    
    test_script = f"""
    const fs = require('fs');
    let code = fs.readFileSync('{js_path}', 'utf8');
    // Mock window and document
    global.window = {{}};
    global.document = {{ addEventListener: () => {{}} }};
    eval(code);

    const questions = [
        ["What is this platform?", "platform_overview", "HybridQML"],
        ["what does this do", "platform_overview", "disease detection"],
        ["What is quantum machine learning?", "what_is_qml", "Hilbert"],
        ["what is QML", "what_is_qml", "Hilbert"],
        ["Why is the quantum model less accurate than classical?", "quantum_vs_classical", "30 raw"],
        ["why is classical better", "quantum_vs_classical", "30 raw"],
        ["Is this using real patient data?", "privacy", "Breast Cancer Wisconsin"],
        ["Is this a certified medical device?", "disclaimer", "not certified"],
        ["How do I try a prediction?", "try_predict", "Try it"],
        ["What is the parameter-shift rule?", "param_shift", "parameter-shift"],
        ["Who built this?", "team", "TEAM PLATINUM"],
        ["who is the team", "team", "TEAM PLATINUM"],
        ["How do I upload patient data?", "upload", "Upload Batch"],
        ["What quantum backends do you support?", "backends", "NumPy Simulator"],
        ["Hello there!", "greetings", "HybridQML Platform Assistant"],
        ["Random gibberish question xyz 12345?", "fallback", "I'm not sure about that one"]
    ];

    const results = [];
    for (const [q, label, snippet] of questions) {{
        const ans = global.window.HybridQMLAssistant.matchIntent(q);
        const contains = ans && ans.includes(snippet);
        results.push({{ q, label, snippet, ok: contains, ans: ans ? ans.substring(0, 60) : null }});
    }}
    console.log(JSON.stringify(results));
    """
    
    res = subprocess.run(["node", "-e", test_script], capture_output=True, text=True)
    assert res.returncode == 0, f"Node execution error: {res.stderr}"
    
    results = json.loads(res.stdout)
    for item in results:
        assert item["ok"], f"Failed query '{item['q']}': Expected snippet '{item['snippet']}', got '{item['ans']}'"
