"""
app.py
------
Flask dashboard for the SIH26139 Hybrid Quantum ML Platform prototype
with secure User Authentication, Role-Based Access Control, Persistent
Prediction Audit Trail, and SQLite storage.

Routes:
  GET  /            Landing overview page (Public)
  GET  /login       User sign in page (Public)
  POST /login       Processes user authentication (Public)
  GET  /signup      Practitioner registration page (Public)
  POST /signup      Processes new account creation (Public)
  GET  /logout      Terminates user session (Protected)
  GET  /dashboard   Results dashboard with Recent Activity (Protected)
  GET  /predict     Interactive "try a patient" page (Protected)
  POST /predict     Executes & records dual prediction for single patient (Protected)
  GET  /upload      Batch patient data upload & screening page (Protected)
  POST /upload      Executes & records batch patient screening (Protected)
  GET  /history     Filterable & paginated diagnostic history for current user (Protected)
  GET  /history/<id> Full diagnostic breakdown for specific record (Protected)
  GET  /admin       Administrator Control Console (Admin-only Protected)
  GET  /admin/all-history Global cross-account prediction audit log (Admin-only Protected)
  POST /run         Initiates asynchronous background pipeline execution (Protected)
  GET  /progress    Live progress & telemetry page (Protected)
  GET  /run/status  JSON status polling endpoint for training progress (Protected)
  GET  /download-sample  Downloads sample test CSV (Public)

Run:  python3 app.py   then open http://localhost:5000
"""
import os
import json
import time
import secrets
import re
import threading
from datetime import datetime, timezone
from functools import wraps

import numpy as np
from flask import (
    Flask, render_template, redirect, url_for, request, jsonify, flash, Response
)
from flask_login import (
    LoginManager, login_user, logout_user, login_required, current_user
)
from flask_migrate import Migrate
from flask_wtf.csrf import CSRFProtect, CSRFError
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from core.models import db, User, Prediction
from core.pipeline_runner import run_full_pipeline
from core.patient_batch import generate_sample_csv, validate_and_process_csv
from core.pdf_report import generate_prediction_pdf
from core import charts

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "sih26139-hybrid-quantum-secret-key")

# Database Configuration (SQLite at instance/app.db)
os.makedirs(app.instance_path, exist_ok=True)
db_path = os.path.join(app.instance_path, "app.db")
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{db_path}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Initialize Extensions
db.init_app(app)
migrate = Migrate(app, db)
csrf = CSRFProtect(app)
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://"
)

login_manager = LoginManager()
login_manager.login_view = "login_page"
login_manager.login_message = "Please authenticate to access quantum diagnostics and telemetry."
login_manager.login_message_category = "info"
login_manager.init_app(app)


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


def api_key_or_ip():
    """Rate limiting key function prioritizing X-API-Key with IP fallback."""
    return request.headers.get("X-API-Key") or get_remote_address()


def require_api_key(f):
    """Decorator verifying X-API-Key authentication header on REST API routes."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = (request.headers.get("X-API-Key") or "").strip()
        if not api_key:
            return jsonify({
                "status": "error",
                "error": "Unauthorized",
                "message": "Missing required 'X-API-Key' header. Obtain your key from /settings."
            }), 401

        user = User.query.filter_by(api_key=api_key).first()
        if not user:
            return jsonify({
                "status": "error",
                "error": "Unauthorized",
                "message": "Invalid or revoked API key provided."
            }), 401

        request.api_user = user
        return f(*args, **kwargs)

    return decorated_function


def admin_required(f):
    """Decorator to restrict route access strictly to users with the 'admin' role."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != "admin":
            flash("Access denied: System Administrator privileges required.", "error")
            return redirect(url_for("dashboard"))
        return f(*args, **kwargs)
    return decorated_function


# In-memory storage for execution artifacts and background job state
STATE = {
    "run": None,
    "last_run_timestamp": 0,
    "last_upload_results": None
}

JOB = {
    "status": "idle",       # 'idle' | 'running' | 'completed' | 'error'
    "progress": 0,          # 0 to 100
    "message": "Ready to launch quantum pipeline.",
    "logs": [],
    "error": None,
    "backend": "simulator",
    "backend_display": "Local NumPy Simulator (Offline)"
}

JOB_LOCK = threading.Lock()
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)


def _pipeline_worker(epochs, quantum_backend="simulator", ibm_token=None):
    global STATE, JOB
    try:
        backend_labels = {
            "simulator": "Local NumPy Simulator (Offline)",
            "qiskit_simulator": "Qiskit Statevector Simulator",
            "ibm_hardware": "IBM Quantum Hardware (QPU)"
        }
        backend_display = backend_labels.get(quantum_backend, quantum_backend)

        with JOB_LOCK:
            JOB["status"] = "running"
            JOB["progress"] = 0
            JOB["message"] = f"Initializing {backend_display} environment..."
            JOB["logs"] = []
            JOB["error"] = None
            JOB["backend"] = quantum_backend
            JOB["backend_display"] = backend_display

        def _log_callback(msg):
            with JOB_LOCK:
                JOB["logs"].append(f"[{time.strftime('%H:%M:%S')}] {msg}")

        def _progress_callback(pct, msg):
            with JOB_LOCK:
                JOB["progress"] = pct
                JOB["message"] = msg
                JOB["logs"].append(f"[{time.strftime('%H:%M:%S')}] {msg}")

        _progress_callback(2, f"Starting hybrid pipeline ({backend_display}) with {epochs} epochs...")
        
        run_data = run_full_pipeline(
            epochs=epochs,
            quantum_backend=quantum_backend,
            ibm_token=ibm_token,
            log=_log_callback,
            progress_callback=_progress_callback
        )

        _progress_callback(95, "Generating dark-mode telemetry charts...")
        
        # Generate and cache chart PNGs
        charts.metric_comparison_chart(run_data["results"], os.path.join(STATIC_DIR, "comparison.png"))
        charts.training_loss_chart(run_data["loss_history"], os.path.join(STATIC_DIR, "loss.png"))
        charts.importance_chart(run_data["results"]["quantum_explainability"]["importances"],
                                 "Hybrid VQC — Feature Importance (ΔAUC)", charts.QUANTUM,
                                 os.path.join(STATIC_DIR, "importance_quantum.png"))
        charts.importance_chart(run_data["results"]["classical_explainability"]["importances"],
                                 "Classical (Same 4 Features) — Feature Importance (ΔAUC)", charts.PRIMARY,
                                 os.path.join(STATIC_DIR, "importance_classical.png"))
        if "classical_shap_explainability" in run_data["results"]:
            charts.shap_importance_chart(run_data["results"]["classical_shap_explainability"]["importances"],
                                         "Classical Baseline — SHAP Values (Mean |SHAP| Impact)", charts.PRIMARY,
                                         os.path.join(STATIC_DIR, "importance_shap.png"))
        charts.roc_chart(None, run_data["probas"], os.path.join(STATIC_DIR, "roc.png"))

        with JOB_LOCK:
            STATE["run"] = run_data
            STATE["last_run_timestamp"] = int(time.time())
            JOB["status"] = "completed"
            JOB["progress"] = 100
            JOB["message"] = "Pipeline execution and telemetry synthesis complete!"
            JOB["logs"].append(f"[{time.strftime('%H:%M:%S')}] Execution finished successfully on {backend_display}.")

    except Exception as e:
        import traceback
        print(f"\n[PIPELINE WORKER ERROR] Background training exception encountered:\n{traceback.format_exc()}", flush=True)
        with JOB_LOCK:
            JOB["status"] = "error"
            JOB["error"] = str(e)
            JOB["message"] = f"Pipeline execution failed: {str(e)}"
            JOB["logs"].append(f"[{time.strftime('%H:%M:%S')}] FATAL ERROR: {str(e)}")


# ------------------------------------------------------------------------------
# System Health & Uptime
# ------------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health_check():
    """Instant top-level server health and liveness check."""
    return jsonify({
        "status": "ok",
        "service": "sih26139-hybrid-qml-platform",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "training_status": JOB.get("status", "idle"),
        "active_backend": JOB.get("backend_display", "Local NumPy Simulator (Offline)")
    }), 200


# ------------------------------------------------------------------------------
# Authentication Routes
# ------------------------------------------------------------------------------

@app.route("/login", methods=["GET", "POST"])
@limiter.limit("5 per minute", methods=["POST"])
def login_page():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    next_url = request.args.get("next") or request.form.get("next")

    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""

        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            login_user(user, remember=True)
            flash(f"Welcome back, {user.name}!", "success")
            if next_url and next_url.startswith("/"):
                return redirect(next_url)
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid email address or password. Please verify your credentials.", "error")

    return render_template("login.html", next_url=next_url)


@app.route("/signup", methods=["GET", "POST"])
def signup_page():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        name = (request.form.get("name") or "").strip()
        email = (request.form.get("email") or "").strip().lower()
        role = (request.form.get("role") or "doctor").strip().lower()
        password = request.form.get("password") or ""
        confirm_password = request.form.get("confirm_password") or ""

        # Validation rules
        if not name or len(name) < 2:
            flash("Please enter your full name or professional title.", "error")
            return render_template("signup.html")

        email_regex = r"^[\w\.-]+@[\w\.-]+\.\w+$"
        if not re.match(email_regex, email):
            flash("Please provide a valid corporate or clinical email address.", "error")
            return render_template("signup.html")

        if role not in ("doctor", "admin"):
            role = "doctor"

        if len(password) < 8:
            flash("Password must contain at least 8 characters for clinical security.", "error")
            return render_template("signup.html")

        if password != confirm_password:
            flash("Passwords do not match. Please verify and re-enter.", "error")
            return render_template("signup.html")

        # Duplicate email check
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash("An account with this email address is already registered. Please sign in instead.", "warning")
            return redirect(url_for("login_page"))

        # Create new user securely
        new_user = User(name=name, email=email, role=role)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()

        login_user(new_user)
        flash(f"Account successfully registered! Welcome to the platform, {name}.", "success")
        return redirect(url_for("dashboard"))

    return render_template("signup.html")


@app.route("/logout")
@login_required
def logout_api():
    logout_user()
    flash("You have been securely signed out of the quantum medical console.", "info")
    return redirect(url_for("login_page"))


# ------------------------------------------------------------------------------
# Core Platform Routes
# ------------------------------------------------------------------------------

@app.route("/")
def index():
    default_backend = os.environ.get("QUANTUM_BACKEND", "simulator")
    has_ibm_token = bool(os.environ.get("IBM_QUANTUM_TOKEN") or os.environ.get("QISKIT_IBM_TOKEN"))
    return render_template(
        "index.html",
        has_run=STATE["run"] is not None,
        default_backend=default_backend,
        has_ibm_token=has_ibm_token
    )


@app.route("/run", methods=["POST"])
@login_required
def run():
    with JOB_LOCK:
        if JOB["status"] == "running":
            if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.is_json:
                return jsonify({"status": "running", "redirect": url_for("progress_page")})
            return redirect(url_for("progress_page"))

    form_data = request.form if request.form else (request.get_json(silent=True) or {})
    
    epochs = int(form_data.get("epochs", 30))
    epochs = max(10, min(epochs, 80))

    backend = form_data.get("backend") or os.environ.get("QUANTUM_BACKEND", "simulator")
    if backend not in ("simulator", "qiskit_simulator", "ibm_hardware"):
        backend = "simulator"

    ibm_token = form_data.get("ibm_token") or os.environ.get("IBM_QUANTUM_TOKEN") or os.environ.get("QISKIT_IBM_TOKEN")

    # Launch in background thread
    t = threading.Thread(target=_pipeline_worker, args=(epochs, backend, ibm_token), daemon=True)
    t.start()

    if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.is_json:
        return jsonify({"status": "started", "redirect": url_for("progress_page")})
    
    return redirect(url_for("progress_page"))


@app.route("/progress")
@login_required
def progress_page():
    return render_template("progress.html", has_run=STATE["run"] is not None)


@app.route("/run/status")
@login_required
def run_status():
    with JOB_LOCK:
        return jsonify({
            "status": JOB["status"],
            "progress": JOB["progress"],
            "message": JOB["message"],
            "backend": JOB.get("backend", "simulator"),
            "backend_display": JOB.get("backend_display", "Local NumPy Simulator"),
            "logs": JOB["logs"][-20:],
            "error": JOB["error"],
            "has_run": STATE["run"] is not None
        })


@app.route("/dashboard")
@login_required
def dashboard():
    if STATE["run"] is None:
        flash("Please run the pipeline first to generate model benchmark results.", "info")
        return redirect(url_for("index"))
    
    results = STATE["run"]["results"]
    v = str(STATE["last_run_timestamp"] or int(time.time()))

    # Fetch the user's latest 5 predictions for the Recent Activity widget
    recent_predictions = Prediction.query.filter_by(user_id=current_user.id)\
                                         .order_by(Prediction.created_at.desc())\
                                         .limit(5)\
                                         .all()

    return render_template(
        "dashboard.html",
        results=results,
        v=v,
        recent_predictions=recent_predictions
    )


@app.route("/predict")
@login_required
def predict_page():
    if STATE["run"] is None:
        flash("Please run the pipeline first to train the quantum and classical models.", "info")
        return redirect(url_for("index"))
    
    splits = STATE["run"]["splits"]
    n_test = len(splits["quantum"]["X_test"])
    return render_template("predict.html", n_test=n_test)


@app.route("/predict", methods=["POST"])
@login_required
def predict_api():
    if STATE["run"] is None:
        return jsonify({"error": "No model loaded. Please run the pipeline first."}), 400

    payload = request.get_json(silent=True) or {}
    try:
        idx = int(payload.get("index", 0))
    except (ValueError, TypeError):
        idx = 0

    run_data = STATE["run"]
    splits = run_data["splits"]
    vqc = run_data["vqc"]
    fair_model = run_data.get("fair_model")

    n_test = len(splits["quantum"]["X_test"])
    idx = max(0, min(idx, n_test - 1))
    
    x_q = splits["quantum"]["X_test"][idx]
    x_c = splits["classical_same_features"]["X_test"][idx]
    y_true = int(splits["quantum"]["y_test"][idx])

    # Quantum Inference
    q_proba = float(vqc.predict_proba([x_q])[0])
    q_pred = int(q_proba >= 0.5)
    q_conf = round(q_proba if q_pred == 1 else 1.0 - q_proba, 4)

    # Classical Fair Baseline Inference (Same 4 PCA features)
    if fair_model is not None:
        c_proba = float(fair_model.predict_proba([x_c])[0, 1])
        c_pred = int(c_proba >= 0.5)
        c_conf = round(c_proba if c_pred == 1 else 1.0 - c_proba, 4)
        c_label = "Malignant" if c_pred == 1 else "Benign"
        c_correct = bool(c_pred == y_true)
    else:
        c_proba = None
        c_conf = None
        c_label = "—"
        c_correct = None

    # Auto-record prediction in persistent SQLite database
    pred_id = None
    try:
        feature_dict = {f"PC{k+1}": round(float(x_q[k]), 4) for k in range(len(x_q))}
        metadata = {
            "patient_index": idx,
            "quantum_prediction": "Malignant" if q_pred == 1 else "Benign",
            "quantum_prob": round(q_proba, 4),
            "quantum_conf": q_conf,
            "classical_prediction": c_label,
            "classical_prob": round(c_proba, 4) if c_proba is not None else None,
            "classical_conf": c_conf
        }
        pred_record = Prediction(
            user_id=current_user.id,
            dataset_name="Breast Cancer Wisconsin (Diagnostic)",
            model_used="both",
            input_features=json.dumps(feature_dict),
            prediction_result="Malignant" if q_pred == 1 else "Benign",
            confidence=q_conf,
            true_label="Malignant" if y_true == 1 else "Benign",
            metadata_json=json.dumps(metadata)
        )
        db.session.add(pred_record)
        db.session.commit()
        pred_id = pred_record.id
    except Exception as err:
        db.session.rollback()
        print(f"[PREDICTION LOGGING ERROR] Failed to record inference: {err}")

    return jsonify({
        "index": idx,
        "prediction_id": pred_id,
        "true_label": "Malignant (Disease-Positive)" if y_true == 1 else "Benign (Disease-Negative)",
        "true_int": y_true,
        "quantum_prediction": "Malignant" if q_pred == 1 else "Benign",
        "quantum_confidence": q_conf,
        "quantum_raw_score": round(q_proba, 3),
        "quantum_correct": bool(q_pred == y_true),
        "classical_prediction": c_label,
        "classical_confidence": c_conf,
        "classical_raw_score": round(c_proba, 3) if c_proba is not None else None,
        "classical_correct": c_correct,
        "correct": bool(q_pred == y_true)
    })


@app.route("/upload", methods=["GET"])
@login_required
def upload_page():
    if STATE["run"] is None:
        flash("Please run the pipeline first to train the quantum and classical models before screening patient batches.", "info")
        return redirect(url_for("index"))
    
    return render_template("upload.html", results=STATE["last_upload_results"])


@app.route("/upload", methods=["POST"])
@login_required
def upload_api():
    if STATE["run"] is None:
        error_msg = "No trained models loaded. Please run the quantum pipeline first."
        if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.is_json:
            return jsonify({"error": error_msg}), 400
        flash(error_msg, "error")
        return redirect(url_for("index"))

    # Extract CSV content strictly in-memory
    csv_bytes = None
    if "file" in request.files:
        file = request.files["file"]
        if file.filename == "":
            error_msg = "No file selected. Please choose a patient CSV file to upload."
            if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                return jsonify({"error": error_msg}), 400
            flash(error_msg, "error")
            return redirect(url_for("upload_page"))
        csv_bytes = file.read()
    elif request.is_json:
        csv_bytes = request.json.get("csv_data", "").encode("utf-8")

    if not csv_bytes:
        error_msg = "Empty payload received. Please provide a valid CSV file."
        if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.is_json:
            return jsonify({"error": error_msg}), 400
        flash(error_msg, "error")
        return redirect(url_for("upload_page"))

    # Execute in-memory validation and inference (Zero disk persistence)
    ok, res = validate_and_process_csv(csv_bytes, STATE["run"])

    if not ok:
        error_msg = res.get("error", "CSV validation failed.")
        if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.is_json:
            return jsonify({"error": error_msg, "details": res}), 400
        flash(error_msg, "error")
        return redirect(url_for("upload_page"))

    STATE["last_upload_results"] = res

    # Auto-record all batch inferences in persistent SQLite database
    try:
        for r in res.get("rows", []):
            row_meta = {
                "patient_id": r.get("patient_id"),
                "quantum_prediction": r.get("quantum_prediction"),
                "quantum_prob": r.get("quantum_prob"),
                "quantum_conf": r.get("quantum_conf"),
                "classical_prediction": r.get("classical_prediction"),
                "classical_prob": r.get("classical_prob"),
                "classical_conf": r.get("classical_conf"),
                "agreement": r.get("agreement")
            }
            row_feats = {"patient_id": r.get("patient_id")}
            pred_record = Prediction(
                user_id=current_user.id,
                dataset_name="Breast Cancer Wisconsin (Diagnostic)",
                model_used="both",
                input_features=json.dumps(row_feats),
                prediction_result=r.get("quantum_prediction"),
                confidence=float(r.get("quantum_conf", 0.5)),
                true_label=r.get("ground_truth"),
                metadata_json=json.dumps(row_meta)
            )
            db.session.add(pred_record)
        db.session.commit()
    except Exception as err:
        db.session.rollback()
        print(f"[BATCH LOGGING ERROR] Failed to record batch inferences: {err}")

    if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.is_json:
        return jsonify({
            "status": "success",
            "results": res,
            "message": f"Successfully screened {res['summary']['total_samples']} patient records."
        })

    flash(f"Successfully screened {res['summary']['total_samples']} patient records with dual quantum-classical evaluation!", "success")
    return render_template("upload.html", results=res)


@app.route("/upload/sample-csv")
@app.route("/download-sample")
def download_sample_csv():
    csv_content = generate_sample_csv(n_samples=10, include_ground_truth=True)
    return Response(
        csv_content,
        mimetype="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=sample_patients_test_batch.csv",
            "Cache-Control": "no-cache"
        }
    )


# ------------------------------------------------------------------------------
# History & Detail Routes
# ------------------------------------------------------------------------------

@app.route("/history")
@login_required
def history_page():
    page = request.args.get("page", 1, type=int)
    model_filter = (request.args.get("model") or "").strip()
    result_filter = (request.args.get("result") or "").strip()
    sort_order = (request.args.get("sort") or "newest").strip()

    # Query strictly the logged in user's predictions (User Isolation)
    query = Prediction.query.filter_by(user_id=current_user.id)

    if model_filter in ("both", "quantum", "classical"):
        query = query.filter_by(model_used=model_filter)
    if result_filter in ("Malignant", "Benign"):
        query = query.filter_by(prediction_result=result_filter)

    if sort_order == "oldest":
        query = query.order_by(Prediction.created_at.asc())
    elif sort_order == "conf_desc":
        query = query.order_by(Prediction.confidence.desc())
    elif sort_order == "conf_asc":
        query = query.order_by(Prediction.confidence.asc())
    else:
        query = query.order_by(Prediction.created_at.desc())

    pagination = query.paginate(page=page, per_page=20, error_out=False)

    return render_template(
        "history.html",
        pagination=pagination,
        current_model=model_filter,
        current_result=result_filter,
        current_sort=sort_order
    )


@app.route("/history/<int:pred_id>")
@login_required
def history_detail(pred_id):
    pred = db.session.get(Prediction, pred_id)
    if not pred:
        flash("Diagnostic prediction record not found.", "error")
        return redirect(url_for("history_page"))

    # Security check: Doctor can only access their own records. Admin can access any.
    if pred.user_id != current_user.id and not current_user.is_admin:
        flash("Access denied: You cannot view other practitioners' patient records.", "error")
        return redirect(url_for("history_page"))

    return render_template("history_detail.html", pred=pred)


@app.route("/history/<int:pred_id>/pdf")
@login_required
def download_prediction_pdf(pred_id):
    pred = db.session.get(Prediction, pred_id)
    if not pred:
        flash("Diagnostic prediction record not found.", "error")
        return redirect(url_for("history_page"))

    # Security check: Doctor can only access their own records. Admin can access any.
    if pred.user_id != current_user.id and not current_user.is_admin:
        flash("Access denied: You cannot export other practitioners' patient records.", "error")
        return redirect(url_for("history_page"))

    exp_data = STATE["run"]["results"] if STATE.get("run") else None
    practitioner_name = pred.user.name if pred.user else (current_user.name or "Clinical Practitioner")

    pdf_bytes = generate_prediction_pdf(
        prediction=pred,
        practitioner_name=practitioner_name,
        explainability_data=exp_data
    )

    date_str = pred.created_at.strftime('%Y-%m-%d') if pred.created_at else time.strftime('%Y-%m-%d')
    filename = f"report_patient_{pred.id}_{date_str}.pdf"

    return Response(
        pdf_bytes,
        mimetype="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
            "Cache-Control": "no-cache"
        }
    )


# ------------------------------------------------------------------------------
# Administrator Routes
# ------------------------------------------------------------------------------

@app.route("/admin")
@login_required
@admin_required
def admin_page():
    users = User.query.order_by(User.id.asc()).all()
    total_predictions = Prediction.query.count()
    return render_template("admin.html", users=users, total_predictions=total_predictions)


@app.route("/admin/all-history")
@login_required
@admin_required
def admin_all_history_page():
    page = request.args.get("page", 1, type=int)
    doctor_id = (request.args.get("doctor_id") or "").strip()
    result_filter = (request.args.get("result") or "").strip()

    query = Prediction.query

    if doctor_id.isdigit():
        query = query.filter_by(user_id=int(doctor_id))
    if result_filter in ("Malignant", "Benign"):
        query = query.filter_by(prediction_result=result_filter)

    pagination = query.order_by(Prediction.created_at.desc()).paginate(page=page, per_page=20, error_out=False)
    doctors = User.query.order_by(User.name.asc()).all()

    return render_template(
        "admin_all_history.html",
        pagination=pagination,
        doctors=doctors,
        current_doctor_id=doctor_id,
        current_result=result_filter
    )


# ------------------------------------------------------------------------------
# First-Run Database Initialization & Default Admin Account Seeding
# ------------------------------------------------------------------------------

def init_db_and_seed_admin():
    with app.app_context():
        db.create_all()
        # Schema migration check for SQLite
        try:
            from sqlalchemy import text
            db.session.execute(text("ALTER TABLE users ADD COLUMN api_key VARCHAR(64)"))
            db.session.commit()
        except Exception:
            db.session.rollback()

        # Ensure all existing users have an API key
        try:
            users_without_keys = User.query.filter((User.api_key == None) | (User.api_key == "")).all()
            for u in users_without_keys:
                u.generate_api_key()
            if users_without_keys:
                db.session.commit()
        except Exception:
            db.session.rollback()

        admin = User.query.filter_by(role="admin").first()
        if not admin:
            default_admin_email = "admin@quantum.health"
            generated_admin_pw = secrets.token_urlsafe(10)
            admin = User(
                name="System Administrator",
                email=default_admin_email,
                role="admin"
            )
            admin.set_password(generated_admin_pw)
            admin.generate_api_key()
            db.session.add(admin)
            db.session.commit()
            print("\n" + "=" * 70)
            print("[SECURITY AUDIT] INITIAL DEFAULT ADMIN ACCOUNT SEEDED:")
            print(f"  Email:    {default_admin_email}")
            print(f"  Password: {generated_admin_pw}")
            print(f"  API Key:  {admin.api_key}")
            print(f"  Role:     admin")
            print("=" * 70 + "\n")


init_db_and_seed_admin()


# ------------------------------------------------------------------------------
# REST API Layer (v1) - External Hospital / HIS / EHR Interoperability
# ------------------------------------------------------------------------------

@app.route("/api/v1/predict", methods=["POST"])
@csrf.exempt
@limiter.limit("30 per minute", key_func=api_key_or_ip)
@require_api_key
def api_v1_predict():
    """
    REST endpoint for external hospital systems to run real-time dual inference.
    Header: X-API-Key: <key>
    JSON Body: { "features": [0.28, -0.15, 0.08, -0.02], "dataset": "optional" }
    """
    data = request.get_json(silent=True)
    if not data or not isinstance(data, dict):
        return jsonify({
            "status": "error",
            "error": "Bad Request",
            "message": "Malformed JSON payload. Expected JSON object with 'features' key."
        }), 400

    raw_features = data.get("features")
    if raw_features is None:
        return jsonify({
            "status": "error",
            "error": "Bad Request",
            "message": "Missing required field 'features' in JSON payload."
        }), 400

    # Parse and validate feature vector
    try:
        if isinstance(raw_features, dict):
            feat_list = [float(v) for k, v in sorted(raw_features.items())]
        elif isinstance(raw_features, (list, tuple)):
            feat_list = [float(v) for v in raw_features]
        else:
            raise ValueError("Features must be a list of numbers or dict of dimension->value.")

        if len(feat_list) != 4:
            return jsonify({
                "status": "error",
                "error": "Bad Request",
                "message": f"Expected exactly 4 PCA biomarker features for the 4-qubit VQC circuit, got {len(feat_list)}."
            }), 400
    except (ValueError, TypeError) as err:
        return jsonify({
            "status": "error",
            "error": "Bad Request",
            "message": f"Invalid feature values: {str(err)}"
        }), 400

    # Ensure model is initialized
    if STATE.get("run") is None:
        try:
            STATE["run"] = run_full_pipeline(epochs=2, quantum_backend="simulator")
        except Exception as e:
            return jsonify({
                "status": "error",
                "error": "Service Unavailable",
                "message": f"Quantum engine initialization error: {str(e)}"
            }), 503

    vqc = STATE["run"]["vqc"]
    fair_model = STATE["run"].get("fair_model")

    # Quantum inference
    feat_vec = np.array(feat_list, dtype=float)
    q_proba = float(vqc.predict_proba([feat_vec])[0])
    q_pred = int(q_proba >= 0.5)
    q_conf = round(q_proba if q_pred == 1 else 1.0 - q_proba, 4)

    # Classical inference
    if fair_model is not None:
        c_proba = float(fair_model.predict_proba([feat_vec])[0, 1])
        c_pred = int(c_proba >= 0.5)
        c_conf = round(c_proba if c_pred == 1 else 1.0 - c_proba, 4)
    else:
        c_proba, c_pred, c_conf = q_proba, q_pred, q_conf

    dataset_name = data.get("dataset", "Breast Cancer Wisconsin (Diagnostic)")

    # Record prediction in SQLite database under API user
    pred_record = None
    try:
        feature_dict = {f"PC{k+1}": round(feat_list[k], 4) for k in range(len(feat_list))}
        metadata = {
            "source": "api_v1_predict",
            "quantum_prediction": "Malignant" if q_pred == 1 else "Benign",
            "quantum_prob": round(q_proba, 4),
            "quantum_conf": q_conf,
            "classical_prediction": "Malignant" if c_pred == 1 else "Benign",
            "classical_prob": round(c_proba, 4),
            "classical_conf": c_conf
        }
        pred_record = Prediction(
            user_id=request.api_user.id,
            dataset_name=dataset_name,
            model_used="both",
            input_features=json.dumps(feature_dict),
            prediction_result="Malignant" if q_pred == 1 else "Benign",
            confidence=q_conf,
            true_label=None,
            metadata_json=json.dumps(metadata)
        )
        db.session.add(pred_record)
        db.session.commit()
    except Exception as err:
        db.session.rollback()
        print(f"[API PREDICTION LOGGING ERROR]: {err}")

    return jsonify({
        "status": "success",
        "prediction_id": pred_record.id if pred_record else None,
        "quantum": {
            "prediction": "Malignant" if q_pred == 1 else "Benign",
            "probability": round(q_proba, 4),
            "confidence": q_conf
        },
        "classical": {
            "prediction": "Malignant" if c_pred == 1 else "Benign",
            "probability": round(c_proba, 4),
            "confidence": c_conf
        },
        "consensus": bool(q_pred == c_pred),
        "dataset": dataset_name,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }), 200


@app.route("/api/v1/history", methods=["GET"])
@csrf.exempt
@limiter.limit("60 per minute", key_func=api_key_or_ip)
@require_api_key
def api_v1_history():
    """
    REST endpoint to retrieve paginated prediction history for the authenticated API key.
    Query parameters: page (int), per_page (int, max 100), result (string)
    """
    page = request.args.get("page", 1, type=int)
    per_page = min(request.args.get("per_page", 20, type=int), 100)
    result_filter = (request.args.get("result") or "").strip()

    query = Prediction.query.filter_by(user_id=request.api_user.id)
    if result_filter in ("Malignant", "Benign"):
        query = query.filter_by(prediction_result=result_filter)

    paginated = query.order_by(Prediction.created_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )

    items = [
        {
            "id": p.id,
            "dataset": p.dataset_name,
            "model_used": p.model_used,
            "prediction_result": p.prediction_result,
            "confidence": round(p.confidence, 4),
            "true_label": p.true_label,
            "is_correct": p.is_correct,
            "features": p.features_dict,
            "metadata": p.metadata_dict,
            "created_at": p.created_at.strftime("%Y-%m-%d %H:%M:%S") if p.created_at else None
        }
        for p in paginated.items
    ]

    return jsonify({
        "status": "success",
        "page": paginated.page,
        "per_page": paginated.per_page,
        "total": paginated.total,
        "pages": paginated.pages,
        "predictions": items
    }), 200


@app.route("/api/v1/health", methods=["GET"])
@csrf.exempt
def api_v1_health():
    """Public health check & uptime endpoint."""
    has_run = STATE.get("run") is not None
    backend = STATE.get("run", {}).get("quantum_backend", "simulator") if has_run else "simulator"
    return jsonify({
        "status": "healthy",
        "platform": "SIH26139 Hybrid Quantum ML",
        "version": "1.0.0",
        "quantum_engine": "online" if has_run else "standby",
        "quantum_backend": backend,
        "database": "connected",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }), 200


# ------------------------------------------------------------------------------
# Settings & API Key Management Routes
# ------------------------------------------------------------------------------

@app.route("/settings", methods=["GET"])
@login_required
def settings_page():
    """Practitioner preferences & API Key access dashboard."""
    current_user.ensure_api_key()
    db.session.commit()
    return render_template("settings.html")


@app.route("/settings/regenerate-key", methods=["POST"])
@login_required
def regenerate_api_key_route():
    """Revokes existing API key and issues a fresh one."""
    new_key = current_user.generate_api_key()
    db.session.commit()
    flash(f"New API key issued successfully ({new_key[:12]}...). Old key has been revoked.", "success")
    return redirect(url_for("settings_page"))


@app.route("/api/docs")
def api_docs_page():
    """Interactive REST API documentation for hospital integrations."""
    return render_template("api_docs.html")


# ------------------------------------------------------------------------------
# Global & API Error Handlers
# ------------------------------------------------------------------------------

@app.errorhandler(400)
def handle_bad_request(e):
    if request.path.startswith("/api/"):
        return jsonify({
            "status": "error",
            "error": "Bad Request",
            "message": getattr(e, "description", "Malformed request payload.")
        }), 400
    return e

@app.errorhandler(401)
def handle_unauthorized(e):
    if request.path.startswith("/api/"):
        return jsonify({
            "status": "error",
            "error": "Unauthorized",
            "message": getattr(e, "description", "Authentication required via X-API-Key header.")
        }), 401
    return e

@app.errorhandler(404)
def handle_not_found(e):
    if request.path.startswith("/api/"):
        return jsonify({
            "status": "error",
            "error": "Not Found",
            "message": "Requested API resource does not exist."
        }), 404
    return e

@app.errorhandler(429)
def handle_rate_limit(e):
    if request.path.startswith("/api/"):
        return jsonify({
            "status": "error",
            "error": "Rate Limit Exceeded",
            "message": f"Rate limit exceeded: {getattr(e, 'description', 'Too many requests')}."
        }), 429
    flash("Too many attempts. Please slow down and wait a moment before trying again.", "error")
    return redirect(request.referrer or url_for("index"))

@app.errorhandler(CSRFError)
def handle_csrf_error(e):
    if request.path.startswith("/api/"):
        return jsonify({
            "status": "error",
            "error": "Forbidden",
            "message": "CSRF validation error."
        }), 403
    flash(f"Session verification error: {e.description}. Please refresh and submit again.", "error")
    return redirect(request.referrer or url_for("index"))

@app.errorhandler(500)
def handle_server_error(e):
    if request.path.startswith("/api/"):
        return jsonify({
            "status": "error",
            "error": "Internal Server Error",
            "message": "An internal server error occurred. Telemetry has been logged."
        }), 500
    return e


if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "0").lower() in ("1", "true")
    app.run(debug=debug_mode, host="0.0.0.0", port=5000)
