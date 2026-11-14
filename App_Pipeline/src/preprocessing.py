"""Leakage-safe preprocessing for the Pima Indians Diabetes data set."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

FEATURE_COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def validate_input_frame(X: pd.DataFrame) -> pd.DataFrame:
    """Validate inference/training input and return columns in canonical order."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    missing = set(FEATURE_COLUMNS) - set(X.columns)
    extra = set(X.columns) - set(FEATURE_COLUMNS)
    if missing or extra:
        raise ValueError(f"Invalid schema. Missing={sorted(missing)}, extra={sorted(extra)}")
    if "Outcome" in X.columns:
        raise ValueError("Outcome is a target, not an input feature.")
    values = X[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if values.isna().all(axis=None):
        raise ValueError("Input contains no numeric feature values.")
    return values


class ClinicalZeroToNaN(BaseEstimator, TransformerMixin):
    """Treat clinically impossible zeros as missing values, without fitting data."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        frame = validate_input_frame(X).copy()
        frame[ZERO_AS_MISSING] = frame[ZERO_AS_MISSING].replace(0, np.nan)
        return frame


class DataFrameMedianImputer(BaseEstimator, TransformerMixin):
    """Median imputer whose fitted statistics remain inspectable and train-only."""
    def fit(self, X, y=None):
        self.feature_names_in_ = list(X.columns)
        self.imputer_ = SimpleImputer(strategy="median")
        self.imputer_.fit(X)
        self.statistics_ = dict(zip(self.feature_names_in_, self.imputer_.statistics_))
        return self

    def transform(self, X):
        transformed = self.imputer_.transform(X[self.feature_names_in_])
        return pd.DataFrame(transformed, columns=self.feature_names_in_, index=X.index)
