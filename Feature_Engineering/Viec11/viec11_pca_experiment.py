"""Thử nghiệm giảm chiều bằng PCA (Chương 4 của đề cương) trên bộ feature đã chốt.

Chỉ dùng tập Train (Tien_Xu_Li/Viec_7/X_train.csv, y_train.csv) và 5-fold StratifiedKFold.
Không đọc X_test, y_test. Imputer, ratio, scaler và PCA nằm trong Pipeline nên chỉ fit trên fold Train.

Mô hình: SVM RBF (C=0.1, class_weight='balanced', gamma='scale') - cấu hình nhóm đã chốt.
Quy tắc chọn (biên bản chốt 0,03): F1 và ROC-AUC không thấp hơn mức tốt nhất quá 0,03,
trong các phương án đạt điều kiện chọn Recall cao nhất; hòa thì xét F1 rồi ROC-AUC.
"""
import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("viec11_pca_audit.log", encoding='utf-8'), logging.StreamHandler()],
)
logger = logging.getLogger("Viec11_PCA")

TOLERANCE = 0.03
RANDOM_STATE = 42


class GlucoseBmiRatio(BaseEstimator, TransformerMixin):
    """Thêm Glucose_BMI_Ratio sau khi đã impute (bộ 9 feature đã chốt)."""

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        frame = pd.DataFrame(X, columns=self.columns)
        frame["Glucose_BMI_Ratio"] = frame["Glucose"] / frame["BMI"].clip(lower=1e-6)
        return frame.to_numpy()


def build_pipeline(columns, pca):
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("ratio", GlucoseBmiRatio(columns)),
        ("scaler", StandardScaler()),
    ]
    if pca is not None:
        steps.append(("pca", pca))
    steps.append(("model", SVC(C=0.1, gamma="scale", class_weight="balanced", random_state=RANDOM_STATE)))
    return Pipeline(steps)


def main():
    logger.info("BẮT ĐẦU THỬ NGHIỆM PCA (CHỈ TRAIN, 5-FOLD CV)")
    path = "../../Tien_Xu_Li/Viec_7/"
    X_train = pd.read_csv(f"{path}X_train.csv")
    y_train = pd.read_csv(f"{path}y_train.csv").squeeze()
    columns = X_train.columns.tolist()
    logger.info("Train: %s mẫu, %s feature gốc; lớp dương %s", len(X_train), len(columns), int(y_train.sum()))

    candidates = [("Không PCA (9 feature)", None)]
    candidates += [(f"PCA {k} thành phần", PCA(n_components=k, random_state=RANDOM_STATE)) for k in range(2, 10)]
    candidates += [
        ("PCA giữ 90% phương sai", PCA(n_components=0.90, random_state=RANDOM_STATE)),
        ("PCA giữ 95% phương sai", PCA(n_components=0.95, random_state=RANDOM_STATE)),
    ]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["recall", "f1", "roc_auc", "average_precision"]
    rows = []
    for name, pca in candidates:
        logger.info("Đánh giá: %s", name)
        scores = cross_validate(build_pipeline(columns, pca), X_train, y_train, cv=cv, scoring=scoring)
        row = {"Phương_Án": name}
        for key, label in [("recall", "Recall"), ("f1", "F1"), ("roc_auc", "ROC_AUC"), ("average_precision", "PR_AUC")]:
            row[f"{label}_Mean"] = round(float(scores[f"test_{key}"].mean()), 4)
            row[f"{label}_Std"] = round(float(scores[f"test_{key}"].std()), 4)
        rows.append(row)

    results = pd.DataFrame(rows)
    eligible = results[
        (results["F1_Mean"] >= results["F1_Mean"].max() - TOLERANCE)
        & (results["ROC_AUC_Mean"] >= results["ROC_AUC_Mean"].max() - TOLERANCE)
    ]
    best = eligible.sort_values(["Recall_Mean", "F1_Mean", "ROC_AUC_Mean"], ascending=False, kind="stable").index[0]
    results["Đạt_Ràng_Buộc_0_03"] = results.index.isin(eligible.index)
    results["Được_Chọn"] = results.index == best
    results.to_csv("bang_pca_ket_qua.csv", index=False, encoding="utf-8-sig")
    logger.info("Phương án được chọn theo quy tắc Recall-first: %s", results.loc[best, "Phương_Án"])

    # Phương sai giải thích: mô tả trên toàn bộ Train (không dùng nhãn Test).
    prep = Pipeline(build_pipeline(columns, None).steps[:-1]).fit(X_train, y_train)
    pca_full = PCA(random_state=RANDOM_STATE).fit(prep.transform(X_train))
    variance = pd.DataFrame({
        "Thành_Phần": [f"PC{i + 1}" for i in range(len(pca_full.explained_variance_ratio_))],
        "Phương_Sai_Giải_Thích": np.round(pca_full.explained_variance_ratio_, 4),
        "Phương_Sai_Tích_Lũy": np.round(np.cumsum(pca_full.explained_variance_ratio_), 4),
    })
    variance.to_csv("bang_pca_phuong_sai.csv", index=False, encoding="utf-8-sig")
    logger.info("HOÀN THÀNH THỬ NGHIỆM PCA.")


if __name__ == "__main__":
    main()
