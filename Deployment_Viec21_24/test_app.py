from pathlib import Path
import sys
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prediction import load_artifact, build_model_input, validate_patient, predict_patient, BASE_FEATURES

SAMPLE = {
    "Pregnancies": 2,
    "Glucose": 120,
    "BloodPressure": 70,
    "SkinThickness": 20,
    "Insulin": 80,
    "BMI": 30,
    "DiabetesPedigreeFunction": 0.5,
    "Age": 33,
}


def test_artifact_loads():
    artifact = load_artifact()
    assert artifact["scenario"] == "2_Them_Ratio"
    assert artifact["model_name"] == "SVM (RBF)"


def test_exact_feature_order_and_ratio():
    artifact = load_artifact()
    frame = build_model_input(SAMPLE, artifact["features"])
    assert frame.columns.tolist() == artifact["features"]
    assert frame.loc[0, "Glucose_BMI_Ratio"] == pytest.approx(4.0)


def test_zero_as_missing_for_five_columns_only():
    artifact = load_artifact()
    patient = dict(SAMPLE, Pregnancies=0, Glucose=0, BloodPressure=0, SkinThickness=0, Insulin=0, BMI=0)
    frame = build_model_input(patient, artifact["features"])
    assert frame.loc[0, "Pregnancies"] == 0
    assert frame.loc[0, "Glucose"] != 0
    assert frame.loc[0, "Glucose_BMI_Ratio"] != float("inf")


def test_negative_or_out_of_range_rejected():
    patient = dict(SAMPLE, Glucose=-1)
    assert validate_patient(patient)


def test_missing_feature_rejected():
    patient = dict(SAMPLE)
    patient.pop("Age")
    assert validate_patient(patient)


def test_end_to_end_prediction():
    result = predict_patient(SAMPLE)
    assert result["outcome"] in (0, 1)
    assert 0 <= result["probability"] <= 1
    assert result["model_input"].shape == (1, 9)
