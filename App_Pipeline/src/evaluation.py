"""Evaluation and report artifact helpers.

Protocol: Outcome=1 is the positive class. Recall is the primary metric;
Precision, F1, ROC-AUC, PR-AUC (average precision) and Accuracy are supporting.
Everything here only reads Test predictions of an already selected model.
"""
from __future__ import annotations

from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                             recall_score, f1_score, roc_auc_score,
                             average_precision_score, ConfusionMatrixDisplay,
                             RocCurveDisplay, PrecisionRecallDisplay)

POSITIVE_CLASS = 1
PRIMARY_METRIC = "recall"
SUPPORTING_METRICS = ("precision", "f1", "roc_auc", "pr_auc", "accuracy")


def slug(model_name: str) -> str:
    return model_name.lower().replace(" ", "_")


def evaluate_model(model, X_test, y_test, model_name: str, figure_dir: Path) -> dict:
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, predictions, labels=[0, POSITIVE_CLASS])
    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "pr_auc": average_precision_score(y_test, probabilities),
        "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
    }
    figure_dir.mkdir(parents=True, exist_ok=True)
    display = ConfusionMatrixDisplay(cm, display_labels=[0, 1])
    display.plot(cmap="Blues", values_format="d")
    plt.xlabel("Predicted Outcome"); plt.ylabel("True Outcome")
    plt.title(f"Confusion Matrix — {model_name} (Test)")
    plt.tight_layout()
    plt.savefig(figure_dir / f"confusion_matrix_{slug(model_name)}.png", dpi=160)
    plt.close()
    return metrics


def plot_roc_pr_curves(model, X_test, y_test, model_name: str, figure_dir: Path) -> Path:
    """ROC and Precision-Recall curves of the final model on Test (reporting only)."""
    probabilities = model.predict_proba(X_test)[:, 1]
    prevalence = float(pd.Series(y_test).mean())
    fig, (roc_ax, pr_ax) = plt.subplots(1, 2, figsize=(11, 4.5))
    RocCurveDisplay.from_predictions(y_test, probabilities, pos_label=POSITIVE_CLASS, ax=roc_ax, name=model_name)
    roc_ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Chance")
    roc_ax.set_title("ROC curve (Test)"); roc_ax.legend(loc="lower right")
    PrecisionRecallDisplay.from_predictions(y_test, probabilities, pos_label=POSITIVE_CLASS, ax=pr_ax, name=model_name)
    pr_ax.axhline(prevalence, color="k", linestyle="--", linewidth=0.8, label=f"Prevalence = {prevalence:.3f}")
    pr_ax.set_title("Precision-Recall curve (Test)"); pr_ax.legend(loc="upper right")
    fig.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    path = figure_dir / f"roc_pr_curves_{slug(model_name)}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def error_groups(model, X_test, y_test) -> pd.DataFrame:
    """Label each Test row TN/FP/FN/TP and attach the predicted probability."""
    frame = X_test.copy()
    frame["Outcome"] = list(y_test)
    frame["prediction"] = model.predict(X_test)
    frame["diabetes_probability"] = model.predict_proba(X_test)[:, 1]
    names = {(0, 0): "TN", (0, 1): "FP", (1, 0): "FN", (1, 1): "TP"}
    frame["group"] = [names[(int(t), int(p))] for t, p in zip(frame.Outcome, frame.prediction)]
    return frame


def error_profile(groups: pd.DataFrame, feature_columns: list[str],
                  zero_as_missing: list[str]) -> pd.DataFrame:
    """Median feature values per confusion group; zeros in clinical columns are ignored as missing."""
    values = groups[feature_columns + ["diabetes_probability"]].copy()
    values[zero_as_missing] = values[zero_as_missing].where(values[zero_as_missing] != 0)
    profile = values.groupby(groups["group"]).median()
    profile.insert(0, "n", groups.groupby("group").size())
    return profile.reindex([g for g in ("TP", "FN", "TN", "FP") if g in profile.index])


def markdown_table(frame: pd.DataFrame) -> str:
    header = "| group | " + " | ".join(frame.columns) + " |"
    rule = "|" + "---|" * (len(frame.columns) + 1)
    rows = ["| " + str(index) + " | " + " | ".join(f"{v:g}" for v in row) + " |"
            for index, row in zip(frame.index, frame.round(3).values)]
    return "\n".join([header, rule, *rows])


def compare_groups(profile: pd.DataFrame, a: str, b: str, columns: list[str]) -> str:
    if a not in profile.index or b not in profile.index:
        return f"Không đủ mẫu để so sánh {a} với {b}."
    return "; ".join(f"{c}: {a}={round(profile.at[a, c], 3):g} so với {b}={round(profile.at[b, c], 3):g}"
                     for c in columns)


def probability_label_disagreements(groups: pd.DataFrame) -> int:
    """Rows whose predicted class differs from predict_proba >= 0.5 (possible for SVC/Platt scaling)."""
    return int((groups.prediction != (groups.diabetes_probability >= 0.5).astype(int)).sum())


def write_error_analysis(path: Path, metrics: dict, groups: pd.DataFrame, profile: pd.DataFrame,
                         cv_metrics: dict, model_name: str, params: dict) -> None:
    m = metrics
    disagreements = probability_label_disagreements(groups)
    predicted_positive = groups.loc[groups.prediction == 1, "diabetes_probability"]
    threshold_note = (
        f"Nhãn dự đoán lấy từ `predict()` của estimator. Có {disagreements}/{len(groups)} mẫu Test mà nhãn khác với "
        f"quy tắc `predict_proba >= 0.5`; xác suất nhỏ nhất vẫn được dự đoán là 1 là {predicted_positive.min():.3f}. "
        "Với SVM, `predict()` dùng decision function còn xác suất là Platt scaling hiệu chỉnh riêng, nên hai đại lượng "
        "không trùng ngưỡng 0,5. Confusion matrix, Recall, Precision, F1, Accuracy dựa trên `predict()`; ROC-AUC và "
        "PR-AUC dựa trên xác suất."
        if disagreements else
        "Nhãn dự đoán lấy từ `predict()` và trùng với quy tắc `predict_proba >= 0.5` trên toàn bộ Test."
    )
    positives, negatives = m["tp"] + m["fn"], m["tn"] + m["fp"]
    total = positives + negatives
    key_columns = ["Glucose", "BMI", "Age", "diabetes_probability"]
    cv_rows = "\n".join(
        f"| {k} | {cv_metrics[f'mean_cv_{k}']:.4f} ± {cv_metrics[f'std_cv_{k}']:.4f} | {m[k]:.4f} |"
        for k in ("recall", "f1", "roc_auc")
    )
    text = f"""# Đánh giá cuối trên Test và phân tích lỗi

Model: **{model_name}**, tham số `{params}`, được chọn chỉ bằng 5-fold CV trên Train (xem `run_metadata.json`).
Outcome=1 là lớp dương. Dùng ngưỡng mặc định của estimator; không tinh chỉnh ngưỡng trên Test.

{threshold_note}

## Metric trên Test (n={total}: {positives} dương, {negatives} âm)

| Metric | Giá trị |
|---|---|
| Recall (metric chính) | {m['recall']:.4f} |
| Precision | {m['precision']:.4f} |
| F1 | {m['f1']:.4f} |
| ROC-AUC | {m['roc_auc']:.4f} |
| PR-AUC (average precision) | {m['pr_auc']:.4f} (mức ngẫu nhiên ≈ tỷ lệ dương {positives / total:.3f}) |
| Accuracy | {m['accuracy']:.4f} |

Confusion matrix: TN={m['tn']}, FP={m['fp']}, FN={m['fn']}, TP={m['tp']}.
Hình: `figures/confusion_matrix_{slug(model_name)}.png`, `figures/roc_pr_curves_{slug(model_name)}.png`.

## Train-CV so với Test (chỉ để báo cáo, không dùng để chọn lại)

| Metric | Train-CV mean ± std | Test |
|---|---|---|
{cv_rows}

## False Negative / False Positive

- FN={m['fn']}: bỏ sót {m['fn']}/{positives} mẫu dương ({m['fn'] / positives:.1%}). Đây là lỗi protocol Recall ưu tiên giảm.
- FP={m['fp']}: báo dương {m['fp']}/{negatives} mẫu âm ({m['fp'] / negatives:.1%}); hệ quả là kiểm tra thêm không cần thiết.
- Đánh đổi: trong {m['tp'] + m['fp']} lần báo dương có {m['fp']} lần sai (Precision={m['precision']:.3f}).

Trung vị feature theo nhóm (0 ở Glucose, BloodPressure, SkinThickness, Insulin, BMI được bỏ qua như giá trị thiếu):

{markdown_table(profile)}

- FN so với TP: {compare_groups(profile, 'FN', 'TP', key_columns)}.
- FP so với TN: {compare_groups(profile, 'FP', 'TN', key_columns)}.

Các so sánh trên chỉ mô tả {total} mẫu Test, nhóm FN/FP rất nhỏ; không phải quan hệ nhân quả.

## Giới hạn

- Test chỉ có {total} mẫu ({positives} dương): mỗi FN làm Recall thay đổi khoảng {1 / positives:.3f}, nên ước lượng có sai số lớn.
- Dữ liệu là một quần thể hẹp (phụ nữ gốc Pima, từ 21 tuổi), nhiều giá trị thiếu ở Insulin/SkinThickness; không chứng minh khả năng tổng quát hóa.
- Kết quả phục vụ học tập, **không phải chẩn đoán y khoa**.
- Test đã từng được xem trong các lần chạy trước (khi quy trình cũ còn chọn model theo Test), nên không còn là tập chưa từng quan sát.
"""
    path.write_text(text, encoding="utf-8")


def save_json(data: dict, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
