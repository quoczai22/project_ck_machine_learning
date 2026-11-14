"""Train, compare, evaluate, and persist diabetes classifiers without leakage."""
from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import platform
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.dummy import DummyClassifier
import sklearn

from .preprocessing import FEATURE_COLUMNS, ZERO_AS_MISSING, ClinicalZeroToNaN, DataFrameMedianImputer
from .features import ClinicalFeatureEngineer
from .evaluation import (PRIMARY_METRIC, SUPPORTING_METRICS, evaluate_model, plot_roc_pr_curves,
                         error_groups, error_profile, probability_label_disagreements,
                         write_error_analysis, save_json)

ROOT = Path(__file__).resolve().parents[1]
SELECTION_METRIC_ORDER = ("recall", "f1", "roc_auc")
SELECTION_TOLERANCE = 0.03


def load_data(path: Path) -> pd.DataFrame:
    names = FEATURE_COLUMNS + ["Outcome"]
    data = pd.read_csv(path, header=None, names=names)
    if set(data["Outcome"].unique()) - {0, 1}:
        raise ValueError("Outcome must be binary (0/1).")
    return data


def build_pipeline(classifier) -> Pipeline:
    return Pipeline([
        ("zero_to_nan", ClinicalZeroToNaN()),
        ("imputer", DataFrameMedianImputer()),
        ("features", ClinicalFeatureEngineer()),
        ("scaler", StandardScaler()),
        ("classifier", classifier),
    ])


def model_grids() -> dict:
    return {
        "Logistic Regression": (build_pipeline(LogisticRegression(max_iter=3000, random_state=42)), {
            "classifier__C": [0.1, 1.0, 10.0], "classifier__class_weight": [None, "balanced"]}),
        "Random Forest": (build_pipeline(RandomForestClassifier(random_state=42)), {
            "classifier__n_estimators": [200, 400], "classifier__max_depth": [None, 4, 8],
            "classifier__min_samples_leaf": [1, 3], "classifier__class_weight": [None, "balanced"]}),
        "SVM": (build_pipeline(SVC(probability=True, random_state=42)), {
            "classifier__C": [0.1], "classifier__gamma": ["scale"],
            "classifier__class_weight": [None, "balanced"]}),
    }


def select_candidate(cv_summary: pd.DataFrame, tolerance: float = SELECTION_TOLERANCE) -> int:
    """Choose a candidate row using Train-CV columns only (no Test metric is accepted).

    F1 and ROC-AUC must lie within `tolerance` of their best CV value; among the
    eligible rows maximize Recall, ties broken by F1 then ROC-AUC, then row order.
    """
    columns = [f"mean_cv_{m}" for m in SELECTION_METRIC_ORDER]
    if not set(columns) <= set(cv_summary.columns):
        raise ValueError(f"cv_summary must contain {columns}")
    eligible = cv_summary[(cv_summary.mean_cv_f1 >= cv_summary.mean_cv_f1.max() - tolerance) &
                          (cv_summary.mean_cv_roc_auc >= cv_summary.mean_cv_roc_auc.max() - tolerance)]
    return eligible.sort_values(columns, ascending=False, kind="stable").index[0]


def dataset_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_eda(data: pd.DataFrame, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    ax = sns.countplot(data=data, x="Outcome", hue="Outcome", legend=False, palette="Set2")
    ax.set_title("Outcome distribution")
    plt.tight_layout(); plt.savefig(figure_dir / "outcome_distribution.png", dpi=160); plt.close()
    data[FEATURE_COLUMNS].hist(figsize=(14, 10), bins=24, edgecolor="white")
    plt.suptitle("Feature distributions (raw data)", y=1.02)
    plt.tight_layout(); plt.savefig(figure_dir / "feature_distributions.png", dpi=160, bbox_inches="tight"); plt.close()
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(numeric_only=True), cmap="vlag", center=0, annot=True, fmt=".2f")
    plt.title("Correlation heatmap")
    plt.tight_layout(); plt.savefig(figure_dir / "correlation_heatmap.png", dpi=160); plt.close()


def run_training(raw_path: Path | None = None) -> pd.DataFrame:
    raw_path = raw_path or ROOT / "data" / "raw" / "diabetes.csv"
    reports = ROOT / "reports"
    figures = reports / "figures"
    models_dir = ROOT / "models"
    reports.mkdir(exist_ok=True); models_dir.mkdir(exist_ok=True)
    data = load_data(raw_path)
    X, y = data[FEATURE_COLUMNS], data["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fitted = {}
    cv_rows = []
    candidates = []
    for name, (pipeline, grid) in model_grids().items():
        # A single worker keeps runs reliable on constrained student machines.
        scoring = {metric: metric for metric in SELECTION_METRIC_ORDER}
        search = GridSearchCV(
            pipeline, grid, scoring=scoring, cv=cv, n_jobs=1,
            refit=False, return_train_score=False,
        )
        search.fit(X_train, y_train)
        for i, params in enumerate(search.cv_results_["params"]):
            row = {"model": name, "params": params}
            for metric in SELECTION_METRIC_ORDER:
                row[f"mean_cv_{metric}"] = search.cv_results_[f"mean_test_{metric}"][i]
                row[f"std_cv_{metric}"] = search.cv_results_[f"std_test_{metric}"][i]
            candidates.append((row, pipeline))

    cv_summary = pd.DataFrame([row for row, _ in candidates])
    # Selection sees Train-CV results only; X_test/y_test are first touched below.
    selected_index = select_candidate(cv_summary)
    selected_row, final_model = candidates[selected_index]
    final_name = selected_row["model"]
    final_model.set_params(**selected_row["params"])
    final_model.fit(X_train, y_train)
    # Single final evaluation on Test; results are reported, never fed back into selection.
    metrics = evaluate_model(final_model, X_test, y_test, final_name, figures)
    result = pd.DataFrame([metrics])
    plot_roc_pr_curves(final_model, X_test, y_test, final_name, figures)
    groups = error_groups(final_model, X_test, y_test)
    profile = error_profile(groups, FEATURE_COLUMNS, ZERO_AS_MISSING)
    profile.to_csv(reports / "error_profile_test.csv")
    joblib.dump(final_model, models_dir / "diabetes_pipeline.joblib")
    result.to_csv(reports / "model_comparison.csv", index=False)
    cv_summary.to_csv(reports / "gridsearch_cv_results.csv", index=False)
    selected_cv = {k: float(v) for k, v in selected_row.items() if k.startswith(("mean_cv_", "std_cv_"))}
    metadata = {
        "final_model": final_name,
        "final_params": selected_row["params"],
        "feature_order": FEATURE_COLUMNS + ["Glucose_BMI_Ratio"],
        "selected_cv_index": int(selected_index),
        "selected_cv_metrics": selected_cv,
        "metric_protocol": {"positive_class": 1, "primary": PRIMARY_METRIC, "supporting": list(SUPPORTING_METRICS)},
        "selection_rule": f"Train-CV F1 and ROC-AUC within {SELECTION_TOLERANCE} of respective candidate maxima; maximize Recall, ties F1 then ROC-AUC.",
        "selection_tolerance": SELECTION_TOLERANCE,
        "selection_metrics_source": "5-fold stratified cross-validation within training split only",
        "test_metrics_used_for_selection": False,
        "test_fraction": 0.2,
        "test_size": int(len(X_test)),
        "train_size": int(len(X_train)),
        "random_state": 42,
        "cv_folds": 5,
        "positive_class": 1,
        "test_previously_observed": True,
        "threshold_policy": "Class labels from estimator.predict() (for SVC: decision_function > 0, not predict_proba >= 0.5); not tuned on Test",
        "test_predict_vs_proba_0_5_disagreements": probability_label_disagreements(groups),
        "dataset": {
            "file": (lambda r: str(r.relative_to(ROOT)) if r.is_relative_to(ROOT) else r.name)(raw_path.resolve()),
            "rows": int(len(data)),
            "columns": int(data.shape[1]),
            "sha256": dataset_sha256(raw_path),
            "source_url": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
            "download_date": None,
        },
        "zero_as_missing": ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"],
        "runtime": {"python": platform.python_version(), "scikit_learn": sklearn.__version__},
    }
    save_json(metadata, reports / "run_metadata.json")
    write_error_analysis(reports / "error_analysis.md", metrics, groups, profile, selected_cv,
                         final_name, selected_row["params"])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=None)
    args = parser.parse_args()
    print(run_training(args.data).to_string(index=False))
