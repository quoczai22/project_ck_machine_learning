"""Independent checks for learned preprocessing statistics in every CV fold."""
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from src.training import build_pipeline, load_data
from src.preprocessing import FEATURE_COLUMNS


def test_imputer_and_scaler_are_fit_within_each_fold():
    root = Path(__file__).resolve().parents[1]
    data = load_data(root / 'data/raw/diabetes.csv')
    X, _, y, _ = train_test_split(data[FEATURE_COLUMNS], data.Outcome, test_size=0.2, stratify=data.Outcome, random_state=42)
    for fit_idx, valid_idx in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
        train = X.iloc[fit_idx]
        model = build_pipeline(LogisticRegression(max_iter=1000)).fit(train, y.iloc[fit_idx])
        cleaned = model.named_steps['zero_to_nan'].transform(train)
        imputer = model.named_steps['imputer']
        assert np.isclose(imputer.statistics_['Glucose'], cleaned.Glucose.median())
        engineered = model.named_steps['features'].transform(imputer.transform(cleaned))
        scaler = model.named_steps['scaler']
        assert np.allclose(scaler.mean_, engineered.mean().to_numpy())
        before = scaler.mean_.copy()
        model.predict(X.iloc[valid_idx])
        assert np.array_equal(scaler.mean_, before)
