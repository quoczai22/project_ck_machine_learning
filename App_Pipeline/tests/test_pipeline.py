import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from src.preprocessing import FEATURE_COLUMNS, ClinicalZeroToNaN
from src.features import ClinicalFeatureEngineer
from src.training import build_pipeline, load_data
from src.inference import load_model, predict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def raw_data():
    return load_data(ROOT / "data" / "raw" / "diabetes.csv")


def test_zero_as_missing_and_ratio_is_finite():
    row = pd.DataFrame([[1, 0, 0, 0, 0, 0, 0.5, 30]], columns=FEATURE_COLUMNS)
    cleaned = ClinicalZeroToNaN().fit_transform(row)
    assert cleaned[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]].isna().all().all()
    engineered = ClinicalFeatureEngineer().fit_transform(pd.DataFrame([[1, 100, 70, 20, 0, 30, .5, 30]], columns=FEATURE_COLUMNS))
    assert np.isfinite(engineered["Glucose_BMI_Ratio"]).all()
    assert list(engineered.columns) == FEATURE_COLUMNS + ["Glucose_BMI_Ratio"]


def test_outcome_cannot_be_an_input_feature():
    frame = raw_data().iloc[:2].copy()
    with pytest.raises(ValueError, match="extra"):
        ClinicalZeroToNaN().fit_transform(frame)


def test_imputer_is_fit_only_on_training_rows():
    data = raw_data()
    X_train, X_test, y_train, _ = train_test_split(data[FEATURE_COLUMNS], data.Outcome, test_size=0.2, stratify=data.Outcome, random_state=42)
    assert len(X_train) == 614
    assert len(X_test) == 154
    model = build_pipeline(LogisticRegression(max_iter=1000)).fit(X_train, y_train)
    train_median = X_train["Glucose"].replace(0, np.nan).median()
    full_median = data["Glucose"].replace(0, np.nan).median()
    assert model.named_steps["imputer"].statistics_["Glucose"] == train_median
    # Transforming test rows must not update fitted train statistics.
    before = model.named_steps["imputer"].statistics_.copy()
    model.predict(X_test)
    assert model.named_steps["imputer"].statistics_ == before
    assert isinstance(full_median, float)


def test_fitted_steps_ignore_test_rows():
    """Corrupting Test rows must leave every fitted statistic and Train prediction unchanged."""
    data = raw_data()
    X_train, X_test, y_train, _ = train_test_split(data[FEATURE_COLUMNS], data.Outcome, test_size=0.2, stratify=data.Outcome, random_state=42)
    model = build_pipeline(LogisticRegression(max_iter=1000)).fit(X_train, y_train)
    corrupted = pd.concat([X_train, X_test * 1000])
    leaky = build_pipeline(LogisticRegression(max_iter=1000)).fit(corrupted, pd.concat([y_train, y_train.iloc[:len(X_test)]]))
    # Sanity check: fitting on the corrupted rows does change statistics, so the test can detect leakage.
    assert not np.allclose(leaky.named_steps["scaler"].mean_, model.named_steps["scaler"].mean_)
    expected = X_train[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]].replace(0, np.nan).median()
    for column, value in expected.items():
        assert model.named_steps["imputer"].statistics_[column] == value
    engineered = model[:-2].transform(X_train)
    assert np.allclose(model.named_steps["scaler"].mean_, engineered.mean().values)
    train_scores = model.predict_proba(X_train)[:, 1]
    model.predict_proba(X_test * 1000)
    assert np.allclose(model.predict_proba(X_train)[:, 1], train_scores)


def test_prediction_output_and_input_schema():
    data = raw_data()
    model = build_pipeline(LogisticRegression(max_iter=1000)).fit(data[FEATURE_COLUMNS], data.Outcome)
    result = predict(model, data[FEATURE_COLUMNS].head(3))
    assert list(result.columns) == ["prediction", "diabetes_probability"]
    assert result.diabetes_probability.between(0, 1).all()


def test_saved_pipeline_can_be_loaded_and_predicts():
    model_path = ROOT / "models" / "diabetes_pipeline.joblib"
    model = load_model(model_path)
    result = predict(model, raw_data()[FEATURE_COLUMNS].head(1))
    assert result["prediction"].iloc[0] in (0, 1)
