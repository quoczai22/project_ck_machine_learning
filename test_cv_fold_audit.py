
import numpy as np
import pandas as pd

from google.colab import files
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ==========================================
# 1. CHỌN FILE CSV
# ==========================================

print("Vui lòng chọn file pima_indians_zero_as_missing_NaN.csv")

uploaded = files.upload()
DATA_PATH = next(iter(uploaded.keys()))

df = pd.read_csv(DATA_PATH)

print(f"\nĐã đọc file: {DATA_PATH}")
print(f"Kích thước dữ liệu: {df.shape}")


# ==========================================
# 2. CHUẨN BỊ DỮ LIỆU
# ==========================================

ZERO_AS_MISSING_COLUMNS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

X = df.drop(columns=["Class"])
y = df["Class"]


# ==========================================
# 3. TRANSFORMER
# ==========================================

class ZeroToMissingTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[ZERO_AS_MISSING_COLUMNS] = (
            X[ZERO_AS_MISSING_COLUMNS].replace(0, np.nan)
        )
        return X


# ==========================================
# 4. PIPELINE
# ==========================================

def build_pipeline():
    return Pipeline([
        ("zero_to_missing", ZeroToMissingTransformer()),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(random_state=42)),
    ])


# ==========================================
# 5. TEST CROSS-VALIDATION 5 FOLD
# ==========================================

def test_imputer_and_scaler_are_fit_within_each_fold():

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    result = cross_validate(
        build_pipeline(),
        X,
        y,
        cv=cv,
        scoring="accuracy",
        return_estimator=True,
    )

    # Kiểm tra từng Fold
    for fold_number, ((train_indices, valid_indices), fitted_pipeline) in enumerate(
        zip(cv.split(X, y), result["estimator"]),
        start=1
    ):

        X_fold_train = X.iloc[train_indices]

        # Chỉ lấy phần Train của Fold để tính tham số kỳ vọng
        fold_train_clean = ZeroToMissingTransformer().fit_transform(
            X_fold_train
        )

        expected_imputer = SimpleImputer(strategy="median").fit(
            fold_train_clean
        )

        expected_imputed = expected_imputer.transform(
            fold_train_clean
        )

        expected_scaler = StandardScaler().fit(
            expected_imputed
        )

        # So sánh Imputer
        np.testing.assert_allclose(
            fitted_pipeline.named_steps["imputer"].statistics_,
            expected_imputer.statistics_
        )

        # So sánh Scaler
        np.testing.assert_allclose(
            fitted_pipeline.named_steps["scaler"].mean_,
            expected_scaler.mean_
        )

        np.testing.assert_allclose(
            fitted_pipeline.named_steps["scaler"].scale_,
            expected_scaler.scale_
        )

        print(f"FOLD {fold_number} PASSED")

    assert len(result["estimator"]) == 5
    assert len(result["test_score"]) == 5

    print("\n========== CROSS-VALIDATION RESULTS ==========")

    for i, score in enumerate(result["test_score"], start=1):
        print(f"Fold {i} Accuracy: {score:.4f}")

    print(f"\nMean Accuracy: {np.mean(result['test_score']):.4f}")
    print(f"Std Accuracy:  {np.std(result['test_score']):.4f}")

    print("\nTEST 3 PASSED: Imputer và Scaler chỉ fit trên Train của từng Fold.")


# ==========================================
# 6. CHẠY TEST
# ==========================================

print("\n========== BẮT ĐẦU KIỂM TRA ==========\n")

test_imputer_and_scaler_are_fit_within_each_fold()

print("\n========== HOÀN THÀNH ==========")
