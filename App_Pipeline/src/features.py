"""Feature engineering, fitted only as part of a sklearn Pipeline."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class ClinicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add the Glucose/BMI ratio after imputation (team-agreed set: 8 base features + 1 ratio)."""
    def fit(self, X, y=None):
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X):
        frame = X.copy()
        # BMI has been imputed; clip keeps denominator strictly positive.
        safe_bmi = frame["BMI"].clip(lower=1e-6)
        frame["Glucose_BMI_Ratio"] = frame["Glucose"] / safe_bmi
        return frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)
