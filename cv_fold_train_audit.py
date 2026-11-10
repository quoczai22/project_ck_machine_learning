
import numpy as np
import pandas as pd

from google.colab import files

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)
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
print(f"Total samples: {len(df)}")


# ============================================================
# 3. TRAIN/TEST SPLIT
#    Hold out the final Test set.
#    Cross-validation runs only on Train.
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print(f"\nTrain rows for cross-validation: {len(X_train)}")
print(f"Held-out Test rows (not used in CV): {len(X_test)}")


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

        # Random Forest is used for Data Leakage auditing only.
        # It is not the final model of the project.
        ("model", RandomForestClassifier(random_state=42)),
    ])


# ============================================================
# 5. RUN 5-FOLD CROSS-VALIDATION ON TRAIN ONLY
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

cv_result = cross_validate(
    build_pipeline(),
    X_train,
    y_train,
    cv=cv,
    scoring="accuracy",
    return_estimator=True,
    error_score="raise",
)


# ============================================================
# 6. AUDIT EACH FOLD
#    Recalculate Imputer and Scaler using only that Fold's
#    Train rows, then compare with the fitted Pipeline.
# ============================================================

audit_rows = []

fold_splits = list(cv.split(X_train, y_train))

for fold_number, (
    (train_indices, valid_indices),
    fitted_pipeline
) in enumerate(
    zip(fold_splits, cv_result["estimator"]),
    start=1
):

    X_fold_train = X_train.iloc[train_indices]

    # Fit preprocessing independently using Fold Train only.
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

    actual_imputer = fitted_pipeline.named_steps["imputer"]
    actual_scaler = fitted_pipeline.named_steps["scaler"]

    # Check Imputer parameters.
    imputer_train_only = np.allclose(
        actual_imputer.statistics_,
        expected_imputer.statistics_
    )

    # Check Scaler parameters.
    scaler_mean_train_only = np.allclose(
        actual_scaler.mean_,
        expected_scaler.mean_
    )

    scaler_scale_train_only = np.allclose(
        actual_scaler.scale_,
        expected_scaler.scale_
    )

    fold_passed = (
        imputer_train_only
        and scaler_mean_train_only
        and scaler_scale_train_only
    )

    audit_rows.append({
        "Target_Column": TARGET_COLUMN,
        "Fold": fold_number,
        "Fold_Train_Rows": len(train_indices),
        "Fold_Validation_Rows": len(valid_indices),
        "Imputer_Fit_Fold_Train_Only": imputer_train_only,
        "Scaler_Mean_Fit_Fold_Train_Only": scaler_mean_train_only,
        "Scaler_Scale_Fit_Fold_Train_Only": scaler_scale_train_only,
        "Validation_Accuracy": cv_result["test_score"][fold_number - 1],
        "Audit_Status": "PASS" if fold_passed else "FAIL",
    })


audit_log = pd.DataFrame(audit_rows)


# ============================================================
# 7. DISPLAY RESULTS
# ============================================================

print("\n========== 5-FOLD TRAIN AUDIT LOG ==========")

print(audit_log.to_string(index=False))

print(
    f"\nMean CV Accuracy: "
    f"{np.mean(cv_result['test_score']):.4f}"
)

print(
    f"Std CV Accuracy: "
    f"{np.std(cv_result['test_score']):.4f}"
)

all_passed = audit_log["Audit_Status"].eq("PASS").all()


# ============================================================
# 8. SAVE AND DOWNLOAD THE AUDIT LOG
# ============================================================

output_file = "cross_validation_train_audit_log.csv"

audit_log.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)

print(f"\nSaved audit log: {output_file}")

# Download the audit log even if one of the folds fails.
files.download(output_file)


# ============================================================
# 9. FINAL AUDIT RESULT
# ============================================================

if not all_passed:

    raise AssertionError(
        "At least one CV fold failed the audit. "
        "Please inspect the downloaded audit log."
    )

print("\nCROSS-VALIDATION AUDIT COMPLETED: ALL 5 FOLDS PASSED")

print("Target column: Outcome")
print("Model: Random Forest (Audit only)")
print("The held-out Test set was not used during Cross-validation.")
