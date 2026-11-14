import shutil
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.inference import load_model, predict
from src.preprocessing import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"

DEFAULT = {"Pregnancies": 1, "Glucose": 120.0, "BloodPressure": 70.0, "SkinThickness": 20.0,
           "Insulin": 80.0, "BMI": 28.0, "DiabetesPedigreeFunction": 0.5, "Age": 33}
LOWER = {"Pregnancies": 0, "Glucose": 0.0, "BloodPressure": 0.0, "SkinThickness": 0.0,
         "Insulin": 0.0, "BMI": 0.0, "DiabetesPedigreeFunction": 0.0, "Age": 1}
UPPER = {"Pregnancies": 20, "Glucose": 300.0, "BloodPressure": 200.0, "SkinThickness": 100.0,
         "Insulin": 900.0, "BMI": 80.0, "DiabetesPedigreeFunction": 3.0, "Age": 120}


def run_app(values=None, app_path=APP):
    at = AppTest.from_file(str(app_path), default_timeout=60).run()
    if values is not None:
        for widget, column in zip(at.number_input, FEATURE_COLUMNS):
            widget.set_value(values[column])
        at.run()
        at.button[0].click().run()
    return at


def probability_shown(at) -> float:
    return float(at.metric[0].value.strip("%")) / 100


def test_app_renders_eight_inputs_without_errors():
    at = run_app()
    assert not at.exception
    assert len(at.number_input) == len(FEATURE_COLUMNS) == 8


@pytest.mark.parametrize("name,values", [("default", DEFAULT), ("lower bound", LOWER), ("upper bound", UPPER)])
def test_prediction_for_valid_and_boundary_inputs(name, values):
    at = run_app(values)
    assert not at.exception, name
    assert 0.0 <= probability_shown(at) <= 1.0
    assert len(at.success) + len(at.warning) >= 1


def test_app_result_matches_saved_pipeline():
    at = run_app(DEFAULT)
    model = load_model(ROOT / "models" / "diabetes_pipeline.joblib")
    expected = predict(model, pd.DataFrame([DEFAULT])).iloc[0]
    assert abs(probability_shown(at) - expected.diabetes_probability) < 0.0006
    messages = " ".join(w.value for w in list(at.warning) + list(at.success))
    assert f"Outcome = {int(expected.prediction)}" in messages


def test_zero_clinical_values_are_handled_like_missing():
    at = run_app({**DEFAULT, "Glucose": 0.0, "BloodPressure": 0.0, "SkinThickness": 0.0,
                  "Insulin": 0.0, "BMI": 0.0})
    assert not at.exception
    assert 0.0 <= probability_shown(at) <= 1.0


def test_missing_model_shows_error_instead_of_crashing(tmp_path):
    (tmp_path / "app").mkdir()
    shutil.copy(APP, tmp_path / "app" / "streamlit_app.py")
    shutil.copytree(ROOT / "src", tmp_path / "src")
    at = run_app(app_path=tmp_path / "app" / "streamlit_app.py")
    assert not at.exception
    assert len(at.error) == 1
    assert "models" in at.error[0].value or "model" in at.error[0].value.lower()
