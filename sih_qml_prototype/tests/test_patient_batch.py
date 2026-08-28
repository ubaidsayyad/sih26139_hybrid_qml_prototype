import pytest
import io
import pandas as pd
from core.patient_batch import generate_sample_csv, validate_and_process_csv
from core.pipeline_runner import run_full_pipeline

@pytest.fixture(scope="module")
def trained_pipeline():
    return run_full_pipeline(epochs=2, train_subsample=50)

def test_generate_sample_csv():
    csv_str = generate_sample_csv(n_samples=6, include_ground_truth=True)
    df = pd.read_csv(io.StringIO(csv_str))
    assert len(df) == 6
    assert "patient_id" in df.columns
    assert "ground_truth" in df.columns
    assert "mean radius" in df.columns

def test_validate_and_process_csv_success(trained_pipeline):
    csv_str = generate_sample_csv(n_samples=5, include_ground_truth=True)
    ok, res = validate_and_process_csv(csv_str, trained_pipeline)
    assert ok is True
    assert res["summary"]["total_samples"] == 5
    assert len(res["rows"]) == 5
    assert "quantum_prediction" in res["rows"][0]
    assert "classical_prediction" in res["rows"][0]
    assert "agreement" in res["rows"][0]

def test_validate_and_process_csv_missing_columns(trained_pipeline):
    bad_csv = "patient_id,mean radius\nPT-1,14.5\n"
    ok, res = validate_and_process_csv(bad_csv, trained_pipeline)
    assert ok is False
    assert "Missing" in res["error"]

def test_validate_and_process_csv_non_numeric(trained_pipeline):
    csv_str = generate_sample_csv(n_samples=3)
    lines = csv_str.splitlines()
    header = lines[0].split(',')
    idx = header.index("mean radius")
    row1 = lines[1].split(',')
    row1[idx] = "NON_NUMERIC_STR"
    corrupt_csv = "\n".join([lines[0], ",".join(row1)] + lines[2:])
    
    ok, res = validate_and_process_csv(corrupt_csv, trained_pipeline)
    assert ok is False
    assert "Invalid non-numeric" in res["error"]
