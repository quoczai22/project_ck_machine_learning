import numpy as np
import pandas as pd

from google.colab import files
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. CHOOSE THE CSV FILE
# ============================================================
print("Choose pima_indians_zero_as_missing_NaN.csv")
uploaded = files.upload()
if not uploaded:
    raise RuntimeError("No CSV file was selected.")

file_name = next(iter(uploaded))
df = pd.read_csv(file_name)
print(f"\nLoaded: {file_name}")
print(f"Dataset shape: {df.shape}")


# ============================================================
# 2. VALIDATE COLUMNS AND DEFINE X, y
#    The group's dataset uses 'Class' as the target column.
# ============================================================
TARGET_COLUMN = "Class"
ZERO_AS_MISSING_COLUMNS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

if TARGET_COLUMN not in df.columns:
    raise ValueError(
        f"Target column '{TARGET_COLUMN}' was not found. "
        f"Available columns: {df.columns.tolist()}"
    )

missing_features = [
    col for col in ZERO_AS_MISSING_COLUMNS if col not in df.columns
]
if missing_features:
    raise ValueError(f"Missing required feature columns: {missing_features}")

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

# Keep the same 80/20 split used by the project.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print(f"Target column: {TARGET_COLUMN}")
print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")


# ============================================================
# 3. PREPROCESSING TRANSFORMER AND PIPELINE
# ============================================================
class ZeroToMissingTransformer(BaseEstimator, TransformerMixin):
    """Replace zero with NaN in the five specified medical features."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[ZERO_AS_MISSING_COLUMNS] = X[ZERO_AS_MISSING_COLUMNS].replace(
            0, np.nan
        )
        return X


def build_pipeline():
    return Pipeline([
        ("zero_to_missing", ZeroToMissingTransformer()),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(random_state=42)),
    ])


# ============================================================
# 4. FIT THE PIPELINE ON TRAIN ONLY
# ============================================================
pipeline = build_pipeline()
pipeline.fit(X_train, y_train)

# Independently calculate the expected preprocessing parameters
# using only the training rows, then compare them with the pipeline.
X_train_clean = ZeroToMissingTransformer().fit_transform(X_train)
expected_imputer = SimpleImputer(strategy="median").fit(X_train_clean)
X_train_imputed = expected_imputer.transform(X_train_clean)
expected_scaler = StandardScaler().fit(X_train_imputed)

actual_imputer = pipeline.named_steps["imputer"]
actual_scaler = pipeline.named_steps["scaler"]

imputer_train_only = np.allclose(
    actual_imputer.statistics_, expected_imputer.statistics_
)
scaler_mean_train_only = np.allclose(
    actual_scaler.mean_, expected_scaler.mean_
)
scaler_scale_train_only = np.allclose(
    actual_scaler.scale_, expected_scaler.scale_
)


# ============================================================
# 5. CHECK THAT TEST PREDICTION DOES NOT REFIT PREPROCESSING
# ============================================================
imputer_before_predict = actual_imputer.statistics_.copy()
scaler_mean_before_predict = actual_scaler.mean_.copy()
scaler_scale_before_predict = actual_scaler.scale_.copy()

test_predictions = pipeline.predict(X_test)

test_prediction_does_not_refit = (
    np.allclose(actual_imputer.statistics_, imputer_before_predict)
    and np.allclose(actual_scaler.mean_, scaler_mean_before_predict)
    and np.allclose(actual_scaler.scale_, scaler_scale_before_predict)
)

test_accuracy = accuracy_score(y_test, test_predictions)


# ============================================================
# 6. CREATE AND DISPLAY TRAIN/TEST AUDIT LOG
# ============================================================
checks = [
    imputer_train_only,
    scaler_mean_train_only,
    scaler_scale_train_only,
    test_prediction_does_not_refit,
]

status = "PASS" if all(checks) else "FAIL"

audit_log = pd.DataFrame([{
    "Target_Column": TARGET_COLUMN,
    "Total_Rows": len(df),
    "Train_Rows": len(X_train),
    "Test_Rows": len(X_test),
    "Imputer_Fit_Train_Only": imputer_train_only,
    "Scaler_Mean_Fit_Train_Only": scaler_mean_train_only,
    "Scaler_Scale_Fit_Train_Only": scaler_scale_train_only,
    "Test_Prediction_Does_Not_Refit": test_prediction_does_not_refit,
    "Test_Accuracy": test_accuracy,
    "Audit_Status": status,
}])

print("\n========== TRAIN/TEST AUDIT LOG ==========")
print(audit_log.to_string(index=False))


# ============================================================
# 7. SAVE AND DOWNLOAD THE AUDIT LOG
# ============================================================
output_file = "train_test_audit_log.csv"
audit_log.to_csv(output_file, index=False)
print(f"\nSaved audit log: {output_file}")
files.download(output_file)

if status != "PASS":
    raise AssertionError("Train/Test audit failed. Check the audit log.")

print("\nTRAIN/TEST AUDIT COMPLETED: ALL CHECKS PASSED")
