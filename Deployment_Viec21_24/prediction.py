"""Inference utilities for the frozen Pima diabetes model.

The selected estimator is loaded from artifacts/pipeline.joblib. This module
only prepares one patient's features and runs inference; it never trains.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Any
import json
import joblib
import numpy as np
import pandas as pd

BASE_FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
DEFAULT_ARTIFACT = Path(__file__).resolve().parent / "artifacts" / "pipeline.joblib"
DEFAULT_METADATA = Path(__file__).resolve().parent / "artifacts" / "metadata.json"


def load_artifact(path: str | Path = DEFAULT_ARTIFACT) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy model artifact: {path}")
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError("Artifact không đúng định dạng: thiếu khóa 'model'.")
    return artifact


def build_model_input(values: Mapping[str, float], feature_order: list[str]) -> pd.DataFrame:
    """Apply exactly the selected scenario 2 feature transformation."""
    missing = [name for name in BASE_FEATURES if name not in values]
    extra = [name for name in values if name not in BASE_FEATURES]
    if missing:
        raise ValueError(f"Thiếu feature: {', '.join(missing)}")
    if extra:
        raise ValueError(f"Feature không được hỗ trợ: {', '.join(extra)}")

    row = {name: float(values[name]) for name in BASE_FEATURES}
    frame = pd.DataFrame([row], columns=BASE_FEATURES)

    # Match Viec 5: zero is treated as missing for five physiological fields.
    frame[ZERO_AS_MISSING] = frame[ZERO_AS_MISSING].replace(0, np.nan)

    # Match Viec 15 scenario 2: add Glucose_BMI_Ratio; zero BMI -> NaN.
    denominator = frame["BMI"].replace(0, np.nan)
    frame["Glucose_BMI_Ratio"] = frame["Glucose"] / denominator
    frame = frame.replace([np.inf, -np.inf], np.nan)

    # Column order is taken from the saved artifact, not recreated by guess.
    if list(frame.columns) != list(feature_order):
        frame = frame.reindex(columns=feature_order)
    if frame.isna().all(axis=1).any():
        raise ValueError("Dữ liệu đầu vào không có giá trị hợp lệ để dự đoán.")
    return frame


def validate_patient(values: Mapping[str, float]) -> list[str]:
    """Return validation messages for clearly invalid or implausible inputs."""
    errors: list[str] = []
    limits = {
        "Pregnancies": (0, 30),
        "Glucose": (0, 400),
        "BloodPressure": (0, 250),
        "SkinThickness": (0, 150),
        "Insulin": (0, 1500),
        "BMI": (0, 100),
        "DiabetesPedigreeFunction": (0, 5),
        "Age": (18, 120),
    }
    for feature, (low, high) in limits.items():
        if feature not in values:
            errors.append(f"Thiếu {feature}.")
            continue
        value = float(values[feature])
        if not np.isfinite(value):
            errors.append(f"{feature} phải là số hữu hạn.")
        elif value < low or value > high:
            errors.append(f"{feature} nằm ngoài khoảng kiểm tra {low}–{high}.")
    return errors


def predict_patient(values: Mapping[str, float], artifact: dict[str, Any] | None = None) -> dict[str, Any]:
    artifact = artifact or load_artifact()
    errors = validate_patient(values)
    if errors:
        raise ValueError(" ".join(errors))
    feature_order = artifact.get("features") or artifact["model"].feature_names_in_.tolist()
    model_input = build_model_input(values, feature_order)
    model = artifact["model"]
    label = int(model.predict(model_input)[0])
    probability = float(model.predict_proba(model_input)[0, 1]) if hasattr(model, "predict_proba") else None
    return {
        "outcome": label,
        "probability": probability,
        "model_input": model_input,
        "missing_fields_imputed": [c for c in ZERO_AS_MISSING if float(values[c]) == 0],
    }
