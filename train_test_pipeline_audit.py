
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
# 2. VALIDATE AND NORMALIZE TARGET COLUMN
#    Use Outcome as the target column.
# ============================================================

if "Class" in df.columns and "Outcome" not in df.columns:

    df = df.rename(columns={"Class": "Outcome"})
    print("Renamed target column: Class -> Outcome")

elif "Class" in df.columns and "Outcome" in df.columns:

    raise ValueError(
        "Dataset contains both Class and Outcome. "
        "Please check the target columns."
    )

elif "Outcome" not in df.columns:

    raise ValueError(
        "Target column Outcome was not found. "
        f"Available columns: {df.columns.tolist()}"
    )


TARGET_COLUMN = "Outcome"

ZERO_AS_MISSING_COLUMNS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

missing_features = [
    col for col in ZERO_AS_MISSING_COLUMNS
    if col not in df.columns
]

if missing_features:
    raise ValueError(
        f"Missing required feature columns: {missing_features}"
    )

if df[TARGET_COLUMN].isna().any():
    raise ValueError("Target column Outcome contains missing values.")

if not set(df[TARGET_COLUMN].unique()).issubset({0, 1}):
    raise ValueError("Outcome must contain only values 0 and 1.")

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

print(f"Target column: {TARGET_COLUMN}")
print(f"Features: {X.shape[1]}")


# ============================================================
# 3. TRAIN/TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print(f"\nTrain rows: {len(X_train)}")
print(f"Test rows: {len(X_test)}")


# ============================================================
# 4. PREPROCESSING TRANSFORMER AND PIPELINE
# ============================================================

class ZeroToMissingTransformer(BaseEstimator, TransformerMixin):
    """Replace zero with NaN in the five specified medical features."""

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

        # Random Forest is used for leakage auditing only.
        # It is not the final model of the project.
        ("model", RandomForestClassifier(random_state=42)),
    ])


# ============================================================
# 5. FIT PIPELINE ON TRAIN ONLY
# ============================================================

pipeline = build_pipeline()

pipeline.fit(X_train, y_train)


# Independently calculate expected preprocessing parameters
# using only the training rows.

X_train_clean = ZeroToMissingTransformer().fit_transform(
    X_train
)

expected_imputer = SimpleImputer(strategy="median").fit(
    X_train_clean
)

X_train_imputed = expected_imputer.transform(
    X_train_clean
)

expected_scaler = StandardScaler().fit(
    X_train_imputed
)


actual_imputer = pipeline.named_steps["imputer"]
actual_scaler = pipeline.named_steps["scaler"]


# Compare Imputer parameters.

imputer_train_only = np.allclose(
    actual_imputer.statistics_,
    expected_imputer.statistics_
)


# Compare Scaler parameters.

scaler_mean_train_only = np.allclose(
    actual_scaler.mean_,
    expected_scaler.mean_
)

scaler_scale_train_only = np.allclose(
    actual_scaler.scale_,
    expected_scaler.scale_
)


# ============================================================
# 6. CHECK THAT TEST PREDICTION DOES NOT REFIT PREPROCESSING
# ============================================================

imputer_before_predict = actual_imputer.statistics_.copy()

scaler_mean_before_predict = actual_scaler.mean_.copy()

scaler_scale_before_predict = actual_scaler.scale_.copy()


# Only predict on Test. Do not call fit or fit_transform.

test_predictions = pipeline.predict(X_test)


test_prediction_does_not_refit = (
    np.allclose(
        actual_imputer.statistics_,
        imputer_before_predict
    )
    and
    np.allclose(
        actual_scaler.mean_,
        scaler_mean_before_predict
    )
    and
    np.allclose(
        actual_scaler.scale_,
        scaler_scale_before_predict
    )
)


test_accuracy = accuracy_score(
    y_test,
    test_predictions
)


# ============================================================
# 7. CREATE AND DISPLAY TRAIN/TEST AUDIT LOG
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
# 8. SAVE AND DOWNLOAD THE AUDIT LOG
# ============================================================

output_file = "train_test_audit_log.csv"

audit_log.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)

print(f"\nSaved audit log: {output_file}")

# Download the log even if the audit fails.
files.download(output_file)


# ============================================================
# 9. FINAL AUDIT RESULT
# ============================================================

if status != "PASS":

    raise AssertionError(
        "Train/Test audit failed. Please check the audit log."
    )


print("\nTRAIN/TEST AUDIT COMPLETED: ALL CHECKS PASSED")

print("Target column: Outcome")
print("Model: Random Forest (Audit only)")
print("Random Forest is not the final project model.")
