
import numpy as np
import pandas as pd

from google.colab import files
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
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
# 2. CẤU HÌNH DỮ LIỆU
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

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# 3. TRANSFORMER VÀ PIPELINE
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


def build_pipeline():
    return Pipeline([
        ("zero_to_missing", ZeroToMissingTransformer()),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(random_state=42)),
    ])


# ==========================================
# 4. TEST 1: IMPUTER CHỈ FIT TRÊN TRAIN
# ==========================================

def test_imputer_is_fit_only_on_training_rows():

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    train_after_zero = ZeroToMissingTransformer().fit_transform(X_train)

    expected_medians = SimpleImputer(strategy="median").fit(
        train_after_zero
    ).statistics_

    actual_medians = pipeline.named_steps["imputer"].statistics_

    np.testing.assert_allclose(actual_medians, expected_medians)

    assert len(X_train) + len(X_test) == len(df)

    print("TEST 1 PASSED: Imputer chỉ fit trên Train.")


# ==========================================
# 5. TEST 2: DỮ LIỆU TEST KHÔNG ẢNH HƯỞNG
#    ĐẾN CÁC THAM SỐ ĐÃ FIT
# ==========================================

def test_fitted_steps_ignore_test_rows():

    pipeline_original = build_pipeline()
    pipeline_original.fit(X_train, y_train)

    # Thay đổi cực lớn dữ liệu Test
    X_test_changed = X_test.copy()
    X_test_changed.loc[:, :] = X_test_changed * 1000 + 999999

    # Fit lại bằng đúng dữ liệu Train
    pipeline_again = build_pipeline()
    pipeline_again.fit(X_train, y_train)

    # So sánh tham số Imputer
    np.testing.assert_allclose(
        pipeline_original.named_steps["imputer"].statistics_,
        pipeline_again.named_steps["imputer"].statistics_
    )

    # So sánh tham số Scaler
    np.testing.assert_allclose(
        pipeline_original.named_steps["scaler"].mean_,
        pipeline_again.named_steps["scaler"].mean_
    )

    np.testing.assert_allclose(
        pipeline_original.named_steps["scaler"].scale_,
        pipeline_again.named_steps["scaler"].scale_
    )

    # So sánh dự đoán trên Train
    np.testing.assert_array_equal(
        pipeline_original.predict(X_train),
        pipeline_again.predict(X_train)
    )

    # Test chỉ transform/predict, không fit lại
    predictions = pipeline_original.predict(X_test_changed)

    assert len(predictions) == len(X_test_changed)

    print("TEST 2 PASSED: Thay đổi Test không làm thay đổi tham số đã fit.")


# ==========================================
# 6. CHẠY TẤT CẢ TEST
# ==========================================

print("\n========== BẮT ĐẦU KIỂM TRA ==========\n")

test_imputer_is_fit_only_on_training_rows()
test_fitted_steps_ignore_test_rows()

print("\n========== TẤT CẢ TEST ĐÃ HOÀN THÀNH ==========")
