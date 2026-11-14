"""Thử nghiệm LDA (Chương 4 — Giảm chiều dữ liệu) trên bộ 9 feature đã chốt.

Chỉ dùng tập Train (Tien_Xu_Li/Viec_7/X_train.csv, y_train.csv), 5-fold StratifiedKFold; không đọc Test.
Imputer, ratio, scaler và LDA nằm trong Pipeline nên chỉ fit trên fold Train.

LDA cho bài toán nhị phân chỉ có tối đa 1 thành phần, nên có hai cách dùng:
  1) LDA làm giảm chiều (1 thành phần) rồi đưa vào SVM đã chốt.
  2) LDA làm bộ phân lớp trực tiếp (mặc định, và với priors cân bằng 0,5/0,5 để so sánh với class_weight='balanced').
Quy tắc chọn: F1 và ROC-AUC không thấp hơn mức tốt nhất quá 0,03, rồi chọn Recall cao nhất.
"""
import logging

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("viec11_lda_audit.log", encoding='utf-8'), logging.StreamHandler()],
)
logger = logging.getLogger("Viec11_LDA")

TOLERANCE = 0.03
RANDOM_STATE = 42


class GlucoseBmiRatio(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        frame = pd.DataFrame(X, columns=self.columns)
        frame["Glucose_BMI_Ratio"] = frame["Glucose"] / frame["BMI"].clip(lower=1e-6)
        return frame.to_numpy()


def build(columns, tail):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("ratio", GlucoseBmiRatio(columns)),
        ("scaler", StandardScaler()),
        *tail,
    ])


def svm():
    return ("model", SVC(C=0.1, gamma="scale", class_weight="balanced", random_state=RANDOM_STATE))


def main():
    logger.info("BẮT ĐẦU THỬ NGHIỆM LDA (CHỈ TRAIN, 5-FOLD CV)")
    path = "../../Tien_Xu_Li/Viec_7/"
    X_train = pd.read_csv(f"{path}X_train.csv")
    y_train = pd.read_csv(f"{path}y_train.csv").squeeze()
    columns = X_train.columns.tolist()

    candidates = [
        ("SVM đã chốt, không giảm chiều (9 feature)", lambda: build(columns, [svm()])),
        ("LDA 1 thành phần + SVM", lambda: build(columns, [("lda", LinearDiscriminantAnalysis(n_components=1)), svm()])),
        ("LDA phân lớp trực tiếp (priors mặc định)", lambda: build(columns, [("model", LinearDiscriminantAnalysis())])),
        ("LDA phân lớp trực tiếp (priors cân bằng 0,5/0,5)",
         lambda: build(columns, [("model", LinearDiscriminantAnalysis(priors=[0.5, 0.5]))])),
    ]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["recall", "f1", "roc_auc", "average_precision"]
    rows = []
    for name, make in candidates:
        logger.info("Đánh giá: %s", name)
        scores = cross_validate(make(), X_train, y_train, cv=cv, scoring=scoring)
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
    results.to_csv("bang_lda_ket_qua.csv", index=False, encoding="utf-8-sig")
    logger.info("Phương án được chọn theo quy tắc Recall-first: %s", results.loc[best, "Phương_Án"])
    logger.info("HOÀN THÀNH THỬ NGHIỆM LDA.")


if __name__ == "__main__":
    main()
