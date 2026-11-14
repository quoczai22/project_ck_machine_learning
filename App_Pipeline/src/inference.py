"""Model loading and schema-validated prediction API."""
from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd
from .preprocessing import validate_input_frame


def load_model(path: str | Path):
    return joblib.load(path)


def predict(model, X: pd.DataFrame) -> pd.DataFrame:
    frame = validate_input_frame(X)
    probability = model.predict_proba(frame)[:, 1]
    prediction = model.predict(frame)
    return pd.DataFrame({"prediction": prediction, "diabetes_probability": probability}, index=frame.index)
