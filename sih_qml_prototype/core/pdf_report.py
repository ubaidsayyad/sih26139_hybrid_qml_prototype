"""
core/pdf_report.py
------------------
Generates clean, professional, single-page printer-friendly PDF clinical diagnostic reports
for individual patient predictions from the Hybrid Quantum ML platform (SIH26139).

Uses ReportLab to produce high-resolution, accessible, printable clinical documents with:
- Platform branding & metadata
- Attending practitioner & patient identification
- Prominent diagnosis determination banner & confidence metric
- Dual quantum vs classical inference breakdown table
- Biomarker feature telemetry & explainability importance attributions
- Regulatory research disclaimer footer
"""
import io
import json
from datetime import datetime, timezone
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically add page numbers and header/footer rules."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 7.5)
        self.setFillColor(colors.HexColor("#64748B"))
        
        # Bottom disclaimer footer line
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.75)
        self.line(36, 38, letter[0] - 36, 38)

        # Footer text
        disclaimer = (
            "PROTOTYPE FOR RESEARCH & DEMONSTRATION PURPOSES ONLY — "
            "NOT A CERTIFIED DIAGNOSTIC TOOL"
        )
        self.drawString(36, 26, disclaimer)
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(letter[0] - 36, 26, page_str)
        self.restoreState()


def generate_prediction_pdf(prediction, practitioner_name="Authorized Clinical Practitioner", explainability_data=None):
    """
    Generates a PDF byte buffer for a given Prediction model instance.

    Args:
        prediction: Prediction instance (or dict-like object with fields)
        practitioner_name: Name of logged-in doctor/admin
        explainability_data: Optional dict containing feature importances

    Returns:
        bytes: Raw PDF content
    """
    buffer = io.BytesIO()
    
    # 0.5 inch margins = 36 points (compact top/bottom to ensure clean 1-page fit)
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=30,
        bottomMargin=44
    )

    styles = getSampleStyleSheet()

    # Custom styles
    c_brand_dark = colors.HexColor("#0F172A")    # Slate 900
    c_brand_cyan = colors.HexColor("#0284C7")    # Deep sky cyan
    c_brand_indigo = colors.HexColor("#4F46E5")  # Indigo accent
    c_text_main = colors.HexColor("#1E293B")     # Slate 800
    c_text_muted = colors.HexColor("#64748B")    # Slate 500
    c_border = colors.HexColor("#E2E8F0")        # Slate 200

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=17,
        leading=20,
        textColor=c_brand_dark
    )

    sub_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=c_text_muted
    )

    h2_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=13,
        textColor=c_brand_dark,
        spaceBefore=6,
        spaceAfter=3
    )

    cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=c_text_main
    )

    cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=c_brand_dark
    )

    cell_mono = ParagraphStyle(
        'TableCellMono',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8.5,
        leading=11,
        textColor=c_brand_dark
    )

    story = []

    # 1. Header Banner
    header_data = [
        [
            Paragraph("<b>HybridQML · Clinical Diagnostic Telemetry</b>", title_style),
            Paragraph("<b>RECORD ID: #" + str(getattr(prediction, 'id', 'TEST')) + "</b>", ParagraphStyle(
                'RecId', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=10, alignment=2, textColor=c_brand_indigo
            ))
        ],
        [
            Paragraph("Hybrid Quantum-Classical ML Diagnostic Platform · SIH26139", sub_style),
            Paragraph(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}", ParagraphStyle(
                'GenDate', parent=styles['Normal'], fontName='Helvetica', fontSize=8, alignment=2, textColor=c_text_muted
            ))
        ]
    ]
    t_header = Table(header_data, colWidths=[380, 160])
    t_header.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(t_header)
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_brand_cyan, spaceBefore=2, spaceAfter=6))

    # 2. Patient Information Section
    story.append(Paragraph("Patient Information", h2_style))

    # Extract patient information from linked PatientRecord or metadata
    patient_record = getattr(prediction, 'patient_record', None)
    metadata_dict = getattr(prediction, 'metadata_dict', {}) or {}
    if not metadata_dict and hasattr(prediction, 'metadata_json') and prediction.metadata_json:
        try:
            metadata_dict = json.loads(prediction.metadata_json) if isinstance(prediction.metadata_json, str) else {}
        except Exception:
            metadata_dict = {}

    patient_name = None
    patient_age = None
    patient_gender = None

    if patient_record:
        patient_name = getattr(patient_record, 'full_name', None)
        patient_age = getattr(patient_record, 'age', None)
        patient_gender = getattr(patient_record, 'gender', None)

    if not patient_name and 'patient_name' in metadata_dict:
        patient_name = metadata_dict.get('patient_name')
    if patient_age is None and 'patient_age' in metadata_dict:
        patient_age = metadata_dict.get('patient_age')
    if not patient_gender and 'patient_gender' in metadata_dict:
        patient_gender = metadata_dict.get('patient_gender')

    sample_idx = metadata_dict.get('sample_index', metadata_dict.get('patient_index', metadata_dict.get('patient_id')))
    if sample_idx is None:
        sample_idx = getattr(prediction, 'id', 0)

    now_utc = datetime.now(timezone.utc)
    created_dt = getattr(prediction, 'created_at', None)
    if created_dt:
        created_str = created_dt.strftime('%Y-%m-%d %H:%M:%S UTC')
    else:
        created_str = now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')

    dataset_str = getattr(prediction, 'dataset_name', None) or 'Breast Cancer Wisconsin (Diagnostic)'
    practitioner_name = str(practitioner_name or 'Authorized Clinical Practitioner')

    if patient_name:
        # Named Patient Format: includes Patient Name, Age, Gender (if provided)
        age_str = f"{patient_age} Years" if patient_age is not None else "Unspecified"
        gender_str = str(patient_gender) if patient_gender else "Unspecified"
        patient_table_data = [
            [
                Paragraph("<b>Patient Name:</b>", cell_style),
                Paragraph(f"<b>{patient_name}</b>", cell_bold),
                Paragraph("<b>Patient Age:</b>", cell_style),
                Paragraph(f"<b>{age_str}</b>", cell_style)
            ],
            [
                Paragraph("<b>Gender:</b>", cell_style),
                Paragraph(gender_str, cell_style),
                Paragraph("<b>Attending Doctor:</b>", cell_style),
                Paragraph(str(practitioner_name), cell_bold)
            ],
            [
                Paragraph("<b>Inference Timestamp:</b>", cell_style),
                Paragraph(created_str, cell_mono),
                Paragraph("<b>Diagnostic Dataset:</b>", cell_style),
                Paragraph(dataset_str, cell_style)
            ]
        ]
    else:
        # Anonymous / Test Sample Format: shows "Sample/Test Record #N", zero blank fields
        patient_table_data = [
            [
                Paragraph("<b>Patient Identification:</b>", cell_style),
                Paragraph(f"<b>Sample/Test Record #{sample_idx}</b>", cell_bold),
                Paragraph("<b>Cohort Profile:</b>", cell_style),
                Paragraph("Anonymous Held-Out Test Cohort", cell_style)
            ],
            [
                Paragraph("<b>Attending Doctor:</b>", cell_style),
                Paragraph(str(practitioner_name), cell_bold),
                Paragraph("<b>Model Architecture:</b>", cell_style),
                Paragraph("4-Qubit VQC + Classical Baseline", cell_style)
            ],
            [
                Paragraph("<b>Inference Timestamp:</b>", cell_style),
                Paragraph(created_str, cell_mono),
                Paragraph("<b>Diagnostic Dataset:</b>", cell_style),
                Paragraph(dataset_str, cell_style)
            ]
        ]

    t_patient = Table(patient_table_data, colWidths=[125, 155, 115, 145])
    t_patient.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
        ('BOX', (0,0), (-1,-1), 0.75, c_border),
        ('INNERGRID', (0,0), (-1,-1), 0.5, c_border),
        ('TOPPADDING', (0,0), (-1,-1), 3.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3.5),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_patient)
    story.append(Spacer(1, 8))

    # 3. Diagnostic Outcome Box
    pred_res = getattr(prediction, 'prediction_result', None) or 'Malignant'
    raw_conf = getattr(prediction, 'confidence', None)
    confidence_val = float(raw_conf if raw_conf is not None else 0.5)
    true_label = getattr(prediction, 'true_label', None)

    is_malignant = (pred_res == "Malignant")
    banner_bg = colors.HexColor("#FFF1F2") if is_malignant else colors.HexColor("#ECFDF5")
    banner_border = colors.HexColor("#E11D48") if is_malignant else colors.HexColor("#059669")
    banner_text_color = colors.HexColor("#9F1239") if is_malignant else colors.HexColor("#065F46")

    match_str = ""
    if true_label:
        match_str = f" · Pathological Ground Truth: <b>{true_label}</b> (" + ("MATCH ✓" if true_label == pred_res else "MISMATCH ✗") + ")"

    outcome_text = (
        f"<font size='13'><b>DIAGNOSIS DETERMINATION: {pred_res.upper()} "
        f"({'Disease-Positive / Critical' if is_malignant else 'Disease-Negative / Normal'})</b></font><br/>"
        f"<font size='9' color='#475569'>Confidence Score: <b>{confidence_val * 100:.1f}%</b>{match_str}</font>"
    )

    t_outcome = Table([[Paragraph(outcome_text, ParagraphStyle('OutStyle', parent=styles['Normal'], textColor=banner_text_color, leading=16))]], colWidths=[540])
    t_outcome.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), banner_bg),
        ('BOX', (0,0), (-1,-1), 1.25, banner_border),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
    ]))
    story.append(t_outcome)
    story.append(Spacer(1, 10))

    # 4. Dual Model Inference Comparison Table
    story.append(Paragraph("Dual Model Inference & Telemetry Breakdown", h2_style))

    meta_dict = getattr(prediction, 'metadata_dict', {}) or {}
    q_prob = meta_dict.get('quantum_prob', confidence_val if is_malignant else round(1.0 - confidence_val, 4))
    q_conf = meta_dict.get('quantum_conf', confidence_val)
    q_pred = meta_dict.get('quantum_prediction', pred_res)

    c_prob = meta_dict.get('classical_prob', '—')
    c_conf = meta_dict.get('classical_conf', '—')
    c_pred = meta_dict.get('classical_prediction', '—')

    c_conf_str = f"{c_conf * 100:.1f}%" if isinstance(c_conf, (int, float)) else str(c_conf)
    c_prob_str = f"{c_prob:.4f}" if isinstance(c_prob, (int, float)) else str(c_prob)

    model_table_data = [
        [
            Paragraph("<b>Model Branch</b>", cell_bold),
            Paragraph("<b>Evaluated Architecture</b>", cell_bold),
            Paragraph("<b>Predicted Output</b>", cell_bold),
            Paragraph("<b>Raw Score / Prob</b>", cell_bold),
            Paragraph("<b>Confidence</b>", cell_bold),
        ],
        [
            Paragraph("<b>Quantum VQC</b>", ParagraphStyle('Q', parent=cell_bold, textColor=colors.HexColor("#0D9488"))),
            Paragraph("4 Qubits · 2 Layers (RY-RZ-CX Entanglement)", cell_style),
            Paragraph(f"<b>{q_pred}</b>", cell_bold),
            Paragraph(f"{float(q_prob):.4f}", cell_mono),
            Paragraph(f"<b>{float(q_conf) * 100:.1f}%</b>", cell_mono),
        ],
        [
            Paragraph("<b>Classical Baseline</b>", ParagraphStyle('C', parent=cell_bold, textColor=colors.HexColor("#4F46E5"))),
            Paragraph("Logistic Regression (Apples-to-Apples PCA)", cell_style),
            Paragraph(f"<b>{c_pred}</b>", cell_bold),
            Paragraph(c_prob_str, cell_mono),
            Paragraph(f"<b>{c_conf_str}</b>", cell_mono),
        ]
    ]

    t_models = Table(model_table_data, colWidths=[110, 180, 95, 85, 70])
    t_models.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F1F5F9")),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_models)
    story.append(Spacer(1, 10))

    # 5. Biomarker Feature Telemetry & Explainability Attribution Table
    story.append(Paragraph("Biomarker Features & Explainability Attribution", h2_style))

    feats_dict = getattr(prediction, 'features_dict', {}) or {}
    if not feats_dict and hasattr(prediction, 'input_features'):
        try:
            feats_dict = json.loads(prediction.input_features) if prediction.input_features else {}
        except Exception:
            feats_dict = {}

    # Gather explainability weights if available
    exp_weights = {}
    if explainability_data and isinstance(explainability_data, dict):
        raw_imp = explainability_data.get("quantum_explainability", {}).get("importances")
        if raw_imp is None:
            raw_imp = explainability_data.get("classical_explainability", {}).get("importances")
        
        if isinstance(raw_imp, dict):
            exp_weights = raw_imp
        elif isinstance(raw_imp, (list, tuple)):
            exp_weights = {f"PC{i+1}": v for i, v in enumerate(raw_imp)}

    feat_table_data = [
        [
            Paragraph("<b>Feature / Latent Dimension</b>", cell_bold),
            Paragraph("<b>Clinical Domain / Type</b>", cell_bold),
            Paragraph("<b>Ingested Value</b>", cell_bold),
            Paragraph("<b>Model Attribution (ΔAUC)</b>", cell_bold),
        ]
    ]

    if feats_dict:
        for idx, (feat_name, val) in enumerate(list(feats_dict.items())[:6]):  # Keep compact for 1 page
            domain = "Quantum Latent Dimension" if 'pc' in feat_name.lower() else "Clinical Biomarker"
            val_str = f"{val:.4f}" if isinstance(val, (int, float)) else str(val)
            imp_val = exp_weights.get(feat_name, exp_weights.get(f"PC{idx+1}", None))
            imp_str = f"+{imp_val:.3f} ΔAUC" if isinstance(imp_val, (int, float)) else "Evaluated"

            feat_table_data.append([
                Paragraph(f"<b>{feat_name}</b>", cell_bold),
                Paragraph(domain, cell_style),
                Paragraph(val_str, cell_mono),
                Paragraph(imp_str, cell_style),
            ])
    else:
        feat_table_data.append([
            Paragraph("Standard Feature Vector", cell_style),
            Paragraph("Synthetic / Standardized PCA Dimensions", cell_style),
            Paragraph("4 Dimensions", cell_mono),
            Paragraph("Active", cell_style),
        ])

    t_feats = Table(feat_table_data, colWidths=[150, 160, 110, 120])
    t_feats.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F1F5F9")),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_feats)
    story.append(Spacer(1, 10))

    # 6. Build document with custom 2-pass numbered canvas
    doc.build(story, canvasmaker=NumberedCanvas)
    buffer.seek(0)
    return buffer.getvalue()
