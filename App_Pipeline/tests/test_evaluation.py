from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score

from src.evaluation import evaluate_model, error_groups, error_profile
from src.training import build_pipeline, load_data, select_candidate, SELECTION_TOLERANCE
from src.preprocessing import FEATURE_COLUMNS, ZERO_AS_MISSING
from src.inference import load_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def split():
    data = load_data(ROOT / "data" / "raw" / "diabetes.csv")
    return train_test_split(data[FEATURE_COLUMNS], data.Outcome, test_size=0.2, stratify=data.Outcome, random_state=42)


def test_metrics_are_consistent_with_confusion_matrix(tmp_path: Path):
    X_train, X_test, y_train, y_test = split()
    assert len(X_train) == 614
    assert len(X_test) == 154
    assert y_test.sum() == 54 and y_train.sum() == 214
    model = build_pipeline(LogisticRegression(max_iter=1000)).fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test, "test", tmp_path)
    predicted = model.predict(X_test)
    cm = confusion_matrix(y_test, predicted)
    assert metrics["tn"] + metrics["fp"] + metrics["fn"] + metrics["tp"] == len(y_test)
    assert np.isclose(metrics["accuracy"], (cm[0, 0] + cm[1, 1]) / cm.sum())
    tn, fp, fn, tp = cm.ravel()
    assert np.isclose(metrics["recall"], tp / (tp + fn))
    assert np.isclose(metrics["precision"], tp / (tp + fp))
    assert np.isclose(metrics["f1"], 2 * tp / (2 * tp + fp + fn))
    probabilities = model.predict_proba(X_test)[:, 1]
    assert np.isclose(metrics["roc_auc"], roc_auc_score(y_test, probabilities))
    assert np.isclose(metrics["pr_auc"], average_precision_score(y_test, probabilities))
    assert (tmp_path / "confusion_matrix_test.png").exists()


def test_error_groups_match_confusion_matrix():
    X_train, X_test, y_train, y_test = split()
    model = build_pipeline(LogisticRegression(max_iter=1000)).fit(X_train, y_train)
    groups = error_groups(model, X_test, y_test)
    tn, fp, fn, tp = confusion_matrix(y_test, model.predict(X_test)).ravel()
    assert groups.group.value_counts().to_dict() == {k: v for k, v in
                                                     {"TN": tn, "FP": fp, "FN": fn, "TP": tp}.items() if v}
    profile = error_profile(groups, FEATURE_COLUMNS, ZERO_AS_MISSING)
    assert profile["n"].sum() == len(X_test)
    assert not (profile[ZERO_AS_MISSING] == 0).any().any()


def cv_frame(rows):
    return pd.DataFrame(rows, columns=["mean_cv_recall", "mean_cv_f1", "mean_cv_roc_auc"])


def test_selection_maximizes_recall_within_f1_and_auc_tolerance():
    frame = cv_frame([
        [0.95, 0.50, 0.70],   # highest recall but F1/AUC far below best -> not eligible
        [0.75, 0.68, 0.84],
        [0.80, 0.67, 0.83],   # eligible and highest recall among eligible
        [0.60, 0.70, 0.85],
    ])
    assert select_candidate(frame) == 2


def test_selection_ties_broken_by_f1_then_auc_then_order():
    frame = cv_frame([[0.8, 0.66, 0.84], [0.8, 0.68, 0.83], [0.8, 0.68, 0.85], [0.8, 0.68, 0.85]])
    assert select_candidate(frame) == 2


def test_selection_rejects_test_metrics():
    with pytest.raises(ValueError):
        select_candidate(pd.DataFrame({"recall": [0.9], "f1": [0.8], "roc_auc": [0.9]}))


def test_saved_artifacts_are_consistent():
    """Model, metadata, CV table and Test metrics must come from the same run."""
    metadata = json.loads((REPORTS / "run_metadata.json").read_text(encoding="utf-8"))
    cv = pd.read_csv(REPORTS / "gridsearch_cv_results.csv")
    final = pd.read_csv(REPORTS / "model_comparison.csv").iloc[0]
    assert metadata["test_metrics_used_for_selection"] is False
    assert metadata["metric_protocol"] == {"positive_class": 1, "primary": "recall",
                                           "supporting": ["precision", "f1", "roc_auc", "pr_auc", "accuracy"]}
    assert (metadata["train_size"], metadata["test_size"], metadata["test_fraction"]) == (614, 154, 0.2)
    # The CV table alone reproduces the selected candidate.
    assert metadata["selection_tolerance"] == SELECTION_TOLERANCE
    index = select_candidate(cv)
    assert index == metadata["selected_cv_index"]
    assert cv.loc[index, "model"] == metadata["final_model"] == final["model"]
    assert cv.loc[index, "params"] == str(metadata["final_params"])
    # The persisted pipeline has the selected hyperparameters and reproduces the reported Test metrics.
    model = load_model(ROOT / "models" / "diabetes_pipeline.joblib")
    params = model.get_params()
    for key, value in metadata["final_params"].items():
        assert params[key] == value
    _, X_test, _, y_test = split()
    tn, fp, fn, tp = confusion_matrix(y_test, model.predict(X_test)).ravel()
    assert (tn, fp, fn, tp) == (final.tn, final.fp, final.fn, final.tp)
    probabilities = model.predict_proba(X_test)[:, 1]
    assert np.isclose(roc_auc_score(y_test, probabilities), final.roc_auc)
    assert np.isclose(average_precision_score(y_test, probabilities), final.pr_auc)
    disagreements = int((model.predict(X_test) != (probabilities >= 0.5)).sum())
    assert metadata["test_predict_vs_proba_0_5_disagreements"] == disagreements
