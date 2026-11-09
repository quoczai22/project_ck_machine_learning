
# ============================================================
# VIEC 14 - STRATIFIED K-FOLD & GRIDSEARCHCV
# Dataset: Pima Indians Diabetes
# Models: Logistic Regression, Random Forest, SVM
# Comparison: class_weight=None vs class_weight="balanced"
# Based on VIEC 13
# Output: Model/Viec14/
# Khong dung __file__, chay truc tiep trong EDA.ipynb
# Khong dung SMOTE mac dinh
# ============================================================

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from IPython.display import display, Image as IPyImage

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. CAU HINH DU AN VA THU MUC
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5
N_JOBS = 1

try:
    CURRENT_DIR = Path(__file__).resolve().parent
except NameError:
    CURRENT_DIR = Path.cwd().resolve()


def find_project_root(start):
    for path in [start, *start.parents]:
        if (path / "Feature_Engineering").is_dir():
            return path
        if (
            (path / "Model").is_dir()
            and (path / "Tien_Xu_Li").is_dir()
        ):
            return path
    return start


PROJECT_ROOT = find_project_root(CURRENT_DIR)

# Neu EDA.ipynb nam o thu muc goc cua project.
OUTPUT_DIR = PROJECT_ROOT / "Model" / "Viec14"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VIEC13_DIR = PROJECT_ROOT / "Model" / "Viec13"

print("=" * 75)
print("VIEC 14 - STRATIFIED K-FOLD & GRIDSEARCHCV")
print("=" * 75)
print("Project root:", PROJECT_ROOT)
print("Viec 13:", VIEC13_DIR)
print("Thu muc output:", OUTPUT_DIR)


# ============================================================
# 2. TIM VA DOC DATASET GOC
# ============================================================

TARGET_NAMES = {"class", "outcome", "diabetes", "target"}

ALIASES = {
    "pregnancies": "Pregnancies",
    "glucose": "Glucose",
    "bloodpressure": "BloodPressure",
    "blood_pressure": "BloodPressure",
    "skinthickness": "SkinThickness",
    "skin_thickness": "SkinThickness",
    "insulin": "Insulin",
    "bmi": "BMI",
    "diabetespedigreefunction": "DiabetesPedigreeFunction",
    "diabetes_pedigree_function": "DiabetesPedigreeFunction",
    "age": "Age",
    "class": "Class",
    "outcome": "Outcome",
    "diabetes": "Diabetes",
    "target": "Target",
}

BASE_FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]


def normalize_name(name):
    return (
        str(name)
        .strip()
        .replace("\ufeff", "")
        .replace(" ", "")
        .replace("-", "_")
        .lower()
    )


def read_csv_robust(path):
    for separator in [None, ",", ";", "\t"]:
        try:
            if separator is None:
                data = pd.read_csv(path, sep=None, engine="python")
            else:
                data = pd.read_csv(path, sep=separator)

            if data.shape[1] > 1:
                data.columns = [
                    str(c).strip().replace("\ufeff", "")
                    for c in data.columns
                ]
                return data
        except Exception:
            pass

    raise ValueError(f"Khong doc duoc CSV: {path}")


def valid_raw_dataset(path):
    try:
        data = read_csv_robust(path)
    except Exception:
        return False

    cols = {normalize_name(c) for c in data.columns}

    required = {
        "pregnancies",
        "glucose",
        "bmi",
        "age",
    }

    if not required.issubset(cols):
        return False

    if not cols.intersection(TARGET_NAMES):
        return False

    # Tranh doc nham bang thong ke count/mean/std.
    if len(data) < 100:
        return False

    target_col = next(
        (
            c for c in data.columns
            if normalize_name(c) in TARGET_NAMES
        ),
        None,
    )

    if target_col is None:
        return False

    target = pd.to_numeric(
        data[target_col],
        errors="coerce",
    ).dropna()

    return (
        len(target) >= 100
        and set(target.unique()).issubset({0, 1})
    )


def find_dataset():
    preferred = [
        PROJECT_ROOT / "Model" / "Tien_Xu_Li"
        / "pima-indians-diabetes.csv",

        PROJECT_ROOT / "Tien_Xu_Li"
        / "pima-indians-diabetes.csv",

        PROJECT_ROOT / "pima-indians-diabetes.csv",

        PROJECT_ROOT / "Model" / "Tien_Xu_Li"
        / "diabetes.csv",

        PROJECT_ROOT / "diabetes.csv",
    ]

    for path in preferred:
        if path.is_file() and valid_raw_dataset(path):
            return path

    excluded = {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".ipynb_checkpoints",
    }

    candidates = []

    for path in PROJECT_ROOT.rglob("*.csv"):
        if any(part.lower() in excluded for part in path.parts):
            continue

        if any(
            word in path.name.lower()
            for word in [
                "thong_ke",
                "summary",
                "describe",
                "classification_report",
                "baseline_model_results",
                "feature_experiment",
                "bang_2",
                "bang_3",
                "bang_4",
            ]
        ):
            continue

        if valid_raw_dataset(path):
            candidates.append(path)

    if candidates:
        candidates.sort(
            key=lambda p: (
                "pima" not in p.name.lower()
                and "diabetes" not in p.name.lower(),
                len(p.parts),
            )
        )
        return candidates[0]

    raise FileNotFoundError(
        "Khong tim thay dataset Pima goc.\n"
        "Kiem tra file pima-indians-diabetes.csv trong "
        "Model/Tien_Xu_Li/."
    )


DATA_PATH = find_dataset()
df = read_csv_robust(DATA_PATH)

rename_map = {}

for col in df.columns:
    normalized = normalize_name(col)
    if normalized in ALIASES:
        rename_map[col] = ALIASES[normalized]

df.rename(columns=rename_map, inplace=True)

TARGET_COL = next(
    (
        c for c in ["Class", "Outcome", "Diabetes", "Target"]
        if c in df.columns
    ),
    None,
)

if TARGET_COL is None:
    raise ValueError(
        f"Khong tim thay cot target: {df.columns.tolist()}"
    )

missing = [
    c for c in BASE_FEATURES
    if c not in df.columns
]

if missing:
    raise ValueError(f"Dataset thieu cot: {missing}")

for col in BASE_FEATURES + [TARGET_COL]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df.dropna(subset=[TARGET_COL], inplace=True)
df = df[df[TARGET_COL].isin([0, 1])].copy()
df[TARGET_COL] = df[TARGET_COL].astype(int)

df = df[BASE_FEATURES + [TARGET_COL]].copy()

# Giong cach xu ly du lieu cua viec 13.
ZERO_AS_MISSING = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

for col in ZERO_AS_MISSING:
    df.loc[df[col] == 0, col] = np.nan

print("\nDataset:", DATA_PATH)
print("Kich thuoc:", df.shape)
print("Target:", TARGET_COL)
print("\nPhan bo target:")
print(df[TARGET_COL].value_counts().sort_index())


# ============================================================
# 3. DOC KET QUA VIEC 13 DE CHON FEATURE SCENARIO
# ============================================================

SCENARIO_DESCRIPTIONS = {
    "1_Baseline": "8 feature goc",
    "2_Them_Ratio": "Them Glucose BMI Ratio",
    "3_Them_Category": "Them Category BMI va Age",
    "4_Tat_Ca_Feature": "Tat ca feature",
    "5_Bo_SkinThickness": "Tat ca feature, bo SkinThickness",
}


def choose_scenario_from_viec13():
    """
    Neu co Bang_2 cua viec 13:
    chon scenario co ROC-AUC trung binh cao nhat.

    Neu khong co file ket qua viec 13:
    su dung Baseline de dam bao code van chay duoc.
    """

    result_path = (
        VIEC13_DIR
        / "Bang_2_Doi_Chieu_Day_Du_5_Thu_Nghiem.csv"
    )

    if not result_path.is_file():
        alternative_paths = [
            VIEC13_DIR / "baseline_model_results.csv",
            VIEC13_DIR / "Bang_3_Tong_Hop_Hieu_Suat_Mo_Hinh.csv",
        ]

        result_path = next(
            (p for p in alternative_paths if p.is_file()),
            None,
        )

    if result_path is None:
        print(
            "\nKhong tim thay CSV ket qua viec 13. "
            "Tam thoi dung Baseline."
        )
        return "1_Baseline"

    try:
        previous = pd.read_csv(result_path)
        previous.columns = [
            str(c).strip() for c in previous.columns
        ]

        scenario_col = next(
            (
                c for c in [
                    "Ten_Thu_Nghiem",
                    "Feature_Scenario",
                    "Scenario",
                ]
                if c in previous.columns
            ),
            None,
        )

        auc_col = next(
            (
                c for c in [
                    "ROC_AUC",
                    "ROC-AUC",
                    "ROC_AUC_Trung_Binh",
                ]
                if c in previous.columns
            ),
            None,
        )

        if scenario_col is None or auc_col is None:
            print(
                "\nFile viec 13 khong co cot scenario/ROC-AUC "
                "phu hop. Dung Baseline."
            )
            return "1_Baseline"

        previous[auc_col] = pd.to_numeric(
            previous[auc_col],
            errors="coerce",
        )

        means = (
            previous.dropna(subset=[auc_col])
            .groupby(scenario_col)[auc_col]
            .mean()
            .sort_values(ascending=False)
        )

        for name in means.index:
            if name in SCENARIO_DESCRIPTIONS:
                print("\nKet qua tham chieu tu viec 13:", result_path)
                print("ROC-AUC trung binh theo scenario:")
                print(means)
                print("Scenario duoc chon:", name)
                return name

    except Exception as exc:
        print("Khong doc duoc ket qua viec 13:", exc)

    return "1_Baseline"


SELECTED_SCENARIO = "2_Them_Ratio"  # scenario chot trong Viec 15 (Recall-first); khong chon theo ROC-AUC


# ============================================================
# 4. FEATURE ENGINEERING GIONG VIEC 13
# ============================================================

def make_features(X_input, scenario):
    Z = X_input[BASE_FEATURES].copy()

    if scenario != "1_Baseline":
        Z["Glucose_BMI_Ratio"] = (
            Z["Glucose"] / Z["BMI"].replace(0, np.nan)
        )

    if scenario in {
        "3_Them_Category",
        "4_Tat_Ca_Feature",
        "5_Bo_SkinThickness",
    }:
        bmi_category = pd.cut(
            Z["BMI"],
            bins=[-np.inf, 18.5, 25, 30, np.inf],
            labels=[
                "Underweight",
                "Normal",
                "Overweight",
                "Obese",
            ],
        )

        age_category = pd.cut(
            Z["Age"],
            bins=[-np.inf, 30, 45, np.inf],
            labels=["Young", "Middle", "Older"],
        )

        Z["BMI_Category"] = pd.Categorical(
            bmi_category,
            categories=[
                "Underweight",
                "Normal",
                "Overweight",
                "Obese",
            ],
        )

        Z["Age_Category"] = pd.Categorical(
            age_category,
            categories=["Young", "Middle", "Older"],
        )

    if scenario in {
        "4_Tat_Ca_Feature",
        "5_Bo_SkinThickness",
    }:
        Z["Glucose_Insulin_Ratio"] = (
            Z["Glucose"] / Z["Insulin"].replace(0, np.nan)
        )

        Z["Age_BMI_Ratio"] = (
            Z["Age"] / Z["BMI"].replace(0, np.nan)
        )

    if scenario == "5_Bo_SkinThickness":
        Z.drop(columns=["SkinThickness"], inplace=True)

    categorical_cols = [
        c for c in ["BMI_Category", "Age_Category"]
        if c in Z.columns
    ]

    if categorical_cols:
        Z = pd.get_dummies(
            Z,
            columns=categorical_cols,
            dtype=int,
        )

    return Z


X = df[BASE_FEATURES].copy()
y = df[TARGET_COL].copy()

# Giu cung random_state va test_size nhu viec 13.
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

X_train = make_features(X_train_raw, SELECTED_SCENARIO)
X_test = make_features(X_test_raw, SELECTED_SCENARIO)

X_test = X_test.reindex(
    columns=X_train.columns,
    fill_value=0,
)

X_train = X_train.apply(pd.to_numeric, errors="coerce")
X_test = X_test.apply(pd.to_numeric, errors="coerce")

print("\nScenario su dung:", SELECTED_SCENARIO)
print("So feature:", X_train.shape[1])
print("So mau train:", len(X_train))
print("So mau test:", len(X_test))


# ============================================================
# 5. STRATIFIED K-FOLD
# ============================================================

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

print("\nStratifiedKFold:")
print("So fold:", N_SPLITS)
print("Shuffle: True")
print("Random state:", RANDOM_STATE)

# Luu thong tin phan bo lop theo tung fold.
fold_rows = []

for fold_no, (train_idx, val_idx) in enumerate(
    cv.split(X_train, y_train),
    start=1,
):
    y_fold_train = y_train.iloc[train_idx]
    y_fold_val = y_train.iloc[val_idx]

    fold_rows.append({
        "Fold": fold_no,
        "Train_Size": len(train_idx),
        "Validation_Size": len(val_idx),
        "Train_Class_0": int((y_fold_train == 0).sum()),
        "Train_Class_1": int((y_fold_train == 1).sum()),
        "Validation_Class_0": int((y_fold_val == 0).sum()),
        "Validation_Class_1": int((y_fold_val == 1).sum()),
        "Train_Class_1_Ratio": float(y_fold_train.mean()),
        "Validation_Class_1_Ratio": float(y_fold_val.mean()),
    })

fold_df = pd.DataFrame(fold_rows)

fold_df.to_csv(
    OUTPUT_DIR / "StratifiedKFold_Phan_Bo_Lop.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\nPhan bo lop theo fold:")
display(fold_df.round(4))


# ============================================================
# 6. DINH NGHIA PIPELINE VA GRID SEARCH
# ============================================================

# class_weight la tham so cua classifier.
# GridSearchCV se thu ca None va "balanced".
# Imputer va scaler duoc fit rieng trong tung fold.
# Khong su dung SMOTE.
MODEL_CONFIGS = {
    "Logistic Regression": {
        "pipeline": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=3000,
                random_state=RANDOM_STATE,
            )),
        ]),
        "param_grid": {
            "model__C": [0.1, 1.0, 10.0],
            "model__solver": ["liblinear"],
            "model__class_weight": [None, "balanced"],
        },
    },

    "Random Forest": {
        "pipeline": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=1,
            )),
        ]),
        "param_grid": {
            "model__n_estimators": [100, 300],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_leaf": [1, 3],
            "model__class_weight": [None, "balanced"],
        },
    },

    "SVM (RBF)": {
        "pipeline": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVC(
                kernel="rbf",
                probability=True,
                random_state=RANDOM_STATE,
            )),
        ]),
        "param_grid": {
            "model__C": [0.1, 1.0, 10.0],
            "model__gamma": ["scale", 0.01, 0.1],
            "model__class_weight": [None, "balanced"],
        },
    },
}


# ============================================================
# 7. CHAY GRIDSEARCHCV CHO 3 MO HINH
# ============================================================

search_results = []
cv_results_all = []
best_estimators = {}
best_params_rows = []
test_predictions = []

for model_name, config in MODEL_CONFIGS.items():

    print("\n" + "=" * 75)
    print("GRID SEARCH:", model_name)
    print("=" * 75)

    search = GridSearchCV(
        estimator=config["pipeline"],
        param_grid=config["param_grid"],
        scoring={
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc",
        },
        refit="recall",  # Recall la metric chinh (bien ban chot 0,03)
        cv=cv,
        n_jobs=N_JOBS,
        verbose=1,
        return_train_score=True,
        error_score="raise",
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    best_estimators[model_name] = best_model

    best_idx = search.best_index_
    cv_data = search.cv_results_

    best_cv_accuracy = cv_data["mean_test_accuracy"][best_idx]
    best_cv_precision = cv_data["mean_test_precision"][best_idx]
    best_cv_recall = cv_data["mean_test_recall"][best_idx]
    best_cv_f1 = cv_data["mean_test_f1"][best_idx]
    best_cv_auc = cv_data["mean_test_roc_auc"][best_idx]

    y_pred = best_model.predict(X_test)

    if hasattr(best_model, "predict_proba"):
        y_score = best_model.predict_proba(X_test)[:, 1]
    else:
        y_score = best_model.decision_function(X_test)

    test_accuracy = accuracy_score(y_test, y_pred)
    test_precision = precision_score(
        y_test, y_pred, zero_division=0
    )
    test_recall = recall_score(
        y_test, y_pred, zero_division=0
    )
    test_f1 = f1_score(
        y_test, y_pred, zero_division=0
    )
    test_auc = roc_auc_score(y_test, y_score)

    best_class_weight = search.best_params_.get(
        "model__class_weight",
        None,
    )

    if best_class_weight is None:
        best_class_weight_label = "None"
    else:
        best_class_weight_label = str(best_class_weight)

    search_results.append({
        "Mo_Hinh": model_name,
        "Scenario": SELECTED_SCENARIO,
        "So_Feature": X_train.shape[1],
        "Best_Class_Weight": best_class_weight_label,
        "Best_CV_Accuracy": best_cv_accuracy,
        "Best_CV_Precision": best_cv_precision,
        "Best_CV_Recall": best_cv_recall,
        "Best_CV_F1": best_cv_f1,
        "Best_CV_ROC_AUC": best_cv_auc,
        "Test_Accuracy": test_accuracy,
        "Test_Precision": test_precision,
        "Test_Recall": test_recall,
        "Test_F1": test_f1,
        "Test_ROC_AUC": test_auc,
        "Best_Params": json.dumps(
            search.best_params_,
            ensure_ascii=False,
            default=str,
        ),
        "Best_Index": int(best_idx),
    })

    best_params_rows.append({
        "Mo_Hinh": model_name,
        "Best_Class_Weight": best_class_weight_label,
        "Best_CV_ROC_AUC": best_cv_auc,
        "Best_Params": json.dumps(
            search.best_params_,
            ensure_ascii=False,
            default=str,
        ),
    })

    # Luu toan bo ket qua tung candidate trong GridSearch.
    model_cv_df = pd.DataFrame(cv_data)
    model_cv_df.insert(0, "Mo_Hinh", model_name)

    cv_results_all.append(model_cv_df)

    test_predictions.append({
        "Mo_Hinh": model_name,
        "y_true": np.asarray(y_test),
        "y_pred": np.asarray(y_pred),
        "y_score": np.asarray(y_score),
    })

    print("\nTham so tot nhat:")
    print(search.best_params_)
    print(f"Best CV ROC-AUC: {best_cv_auc:.4f}")
    print(f"Test ROC-AUC: {test_auc:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")


# ============================================================
# 8. LUU BANG KET QUA GRIDSEARCHCV
# ============================================================

results_df = pd.DataFrame(search_results)

results_df.sort_values(
    by=["Best_CV_Recall", "Best_CV_F1", "Best_CV_ROC_AUC"],
    ascending=[False, False, False],
    inplace=True,
)

results_df.reset_index(drop=True, inplace=True)

results_df.to_csv(
    OUTPUT_DIR / "Viec14_GridSearchCV_Tong_Hop.csv",
    index=False,
    encoding="utf-8-sig",
)

best_params_df = pd.DataFrame(best_params_rows)

best_params_df.to_csv(
    OUTPUT_DIR / "Viec14_Bo_Tham_So_Toi_Uu.csv",
    index=False,
    encoding="utf-8-sig",
)

all_cv_df = pd.concat(
    cv_results_all,
    ignore_index=True,
)

all_cv_df.to_csv(
    OUTPUT_DIR / "Viec14_Toan_Bo_CV_Results.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\nBANG TONG HOP GRIDSEARCHCV")
display(
    results_df[
        [
            "Mo_Hinh",
            "Best_Class_Weight",
            "Best_CV_Accuracy",
            "Best_CV_Recall",
            "Best_CV_F1",
            "Best_CV_ROC_AUC",
            "Test_Accuracy",
            "Test_Recall",
            "Test_F1",
            "Test_ROC_AUC",
        ]
    ].round(4)
)


# ============================================================
# 9. SO SANH CLASS_WEIGHT NONE VA BALANCED
# ============================================================

# Rut ket qua cac candidate tu cv_results_.
weight_comparison_rows = []

for model_name, config in MODEL_CONFIGS.items():

    model_candidates = all_cv_df[
        all_cv_df["Mo_Hinh"] == model_name
    ].copy()

    if model_candidates.empty:
        continue

    def get_weight_label(params_text):
        try:
            params = eval(
                params_text,
                {"__builtins__": {}},
                {"None": None},
            )
            weight = params.get("model__class_weight", None)
            return "None" if weight is None else str(weight)
        except Exception:
            return "Unknown"

    # params trong GridSearchCV la dict.
    model_cv = pd.DataFrame(
        cv_results_all[
            list(MODEL_CONFIGS.keys()).index(model_name)
        ]
    ).copy()

    for _, row in model_cv.iterrows():
        params = row["params"]
        weight = params.get("model__class_weight", None)

        weight_comparison_rows.append({
            "Mo_Hinh": model_name,
            "Class_Weight": (
                "None" if weight is None else str(weight)
            ),
            "CV_Accuracy": row["mean_test_accuracy"],
            "CV_Precision": row["mean_test_precision"],
            "CV_Recall": row["mean_test_recall"],
            "CV_F1": row["mean_test_f1"],
            "CV_ROC_AUC": row["mean_test_roc_auc"],
            "Tham_So": json.dumps(
                params,
                ensure_ascii=False,
                default=str,
            ),
        })

weight_df = pd.DataFrame(weight_comparison_rows)

weight_df.to_csv(
    OUTPUT_DIR / "Viec14_So_Sanh_Class_Weight.csv",
    index=False,
    encoding="utf-8-sig",
)

# Chon candidate tot nhat theo Recall trong moi nhom model/weight.
weight_best_df = (
    weight_df.sort_values(
        by=["CV_Recall", "CV_F1", "CV_ROC_AUC"],
        ascending=[False, False, False],
    )
    .groupby(
        ["Mo_Hinh", "Class_Weight"],
        as_index=False,
    )
    .first()
)

weight_best_df.to_csv(
    OUTPUT_DIR / "Viec14_Class_Weight_Tot_Nhat.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\nSO SANH CLASS_WEIGHT")
display(
    weight_best_df[
        [
            "Mo_Hinh",
            "Class_Weight",
            "CV_Accuracy",
            "CV_Recall",
            "CV_F1",
            "CV_ROC_AUC",
        ]
    ].round(4)
)


# ============================================================
# 10. HAM XUAT BANG THÀNH PNG
# ============================================================

def save_table_png(data, filename, title):
    table_data = data.copy()

    for col in table_data.columns:
        if pd.api.types.is_float_dtype(table_data[col]):
            table_data[col] = table_data[col].map(
                lambda x: f"{x:.4f}" if pd.notna(x) else ""
            )

    table_data = table_data.astype(str)

    n_rows, n_cols = table_data.shape

    fig, ax = plt.subplots(
        figsize=(
            max(13, n_cols * 1.8),
            max(3.5, (n_rows + 2) * 0.5),
        )
    )

    ax.axis("off")
    ax.set_title(
        title,
        fontsize=14,
        fontweight="bold",
        pad=18,
    )

    table = ax.table(
        cellText=table_data.values,
        colLabels=table_data.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.45)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#B7C9D6")

        if row == 0:
            cell.set_facecolor("#24527A")
            cell.set_text_props(
                color="white",
                weight="bold",
            )
        else:
            cell.set_facecolor(
                "#EAF2F8" if row % 2 == 0 else "white"
            )

    fig.tight_layout()

    path = OUTPUT_DIR / filename
    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)
    print("Da luu:", path.name)


save_table_png(
    results_df[
        [
            "Mo_Hinh",
            "Best_Class_Weight",
            "Best_CV_Accuracy",
            "Best_CV_Recall",
            "Best_CV_F1",
            "Best_CV_ROC_AUC",
            "Test_Accuracy",
            "Test_Recall",
            "Test_F1",
            "Test_ROC_AUC",
        ]
    ],
    "Viec14_Bang_Tong_Hop_GridSearchCV.png",
    "VIEC 14 - TONG HOP GRIDSEARCHCV",
)

save_table_png(
    best_params_df,
    "Viec14_Bang_Tham_So_Toi_Uu.png",
    "VIEC 14 - BO THAM SO TOI UU",
)

save_table_png(
    weight_best_df[
        [
            "Mo_Hinh",
            "Class_Weight",
            "CV_Accuracy",
            "CV_Recall",
            "CV_F1",
            "CV_ROC_AUC",
        ]
    ],
    "Viec14_Bang_So_Sanh_Class_Weight.png",
    "SO SANH CLASS_WEIGHT NONE VA BALANCED",
)


# ============================================================
# 11. BIEU DO SO SANH CV RECALL VA ROC-AUC
# ============================================================

# Bieu do Best CV Recall (Uu tien chinh - refit='recall')
fig, ax = plt.subplots(figsize=(10, 6))

plot_df = results_df.sort_values(
    "Best_CV_Recall",
    ascending=True,
)

ax.barh(
    plot_df["Mo_Hinh"],
    plot_df["Best_CV_Recall"],
    color="#d9534f",
)

ax.set_title(
    "Viec 14 - Best Cross-Validation Recall (refit='recall')",
    fontsize=14,
    fontweight="bold",
)

ax.set_xlabel("Mean CV Recall")
ax.set_xlim(0, 1)
ax.grid(axis="x", alpha=0.3)

fig.tight_layout()

fig.savefig(
    str(OUTPUT_DIR / "Viec14_So_Sanh_Best_CV_Recall.png"),
    dpi=200,
    bbox_inches="tight",
)
plt.close(fig)

# Bieu do Best CV ROC-AUC
fig, ax = plt.subplots(figsize=(10, 6))

plot_df_auc = results_df.sort_values(
    "Best_CV_ROC_AUC",
    ascending=True,
)

ax.barh(
    plot_df_auc["Mo_Hinh"],
    plot_df_auc["Best_CV_ROC_AUC"],
    color="#337ab7",
)

ax.set_title(
    "Viec 14 - Best Cross-Validation ROC-AUC",
    fontsize=14,
    fontweight="bold",
)

ax.set_xlabel("Mean CV ROC-AUC")
ax.set_xlim(0, 1)
ax.grid(axis="x", alpha=0.3)

fig.tight_layout()

fig.savefig(
    str(OUTPUT_DIR / "Viec14_So_Sanh_Best_CV_ROC_AUC.png"),
    dpi=200,
    bbox_inches="tight",
)
plt.close(fig)


# ============================================================
# 12. BIEU DO CLASS_WEIGHT NONE VA BALANCED
# ============================================================

for metric_col, metric_label, filename in [
    ("CV_Recall", "Recall", "Viec14_ClassWeight_Recall.png"),
    ("CV_F1", "F1-Score", "Viec14_ClassWeight_F1.png"),
    ("CV_ROC_AUC", "ROC-AUC", "Viec14_ClassWeight_ROC_AUC.png"),
]:

    pivot = weight_best_df.pivot(
        index="Mo_Hinh",
        columns="Class_Weight",
        values=metric_col,
    )

    desired_order = [
        c for c in ["None", "balanced"]
        if c in pivot.columns
    ]

    pivot = pivot.reindex(columns=desired_order)

    fig, ax = plt.subplots(figsize=(10, 6))

    pivot.plot(
        kind="bar",
        ax=ax,
        width=0.7,
    )

    ax.set_title(
        f"So sanh class_weight theo {metric_label}",
        fontsize=14,
        fontweight="bold",
    )

    ax.set_xlabel("Mo hinh")
    ax.set_ylabel(metric_label)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="class_weight")

    plt.xticks(rotation=0)
    fig.tight_layout()

    fig.savefig(
        str(OUTPUT_DIR / filename),
        dpi=200,
        bbox_inches="tight",
    )

    # plt.show()
    plt.close(fig)


# ============================================================
# 13. LUU CLASSIFICATION REPORT TREN TEST SET
# ============================================================

test_report_rows = []

for item in test_predictions:
    report = classification_report(
        item["y_true"],
        item["y_pred"],
        output_dict=True,
        zero_division=0,
    )

    for label, values in report.items():
        if isinstance(values, dict):
            test_report_rows.append({
                "Mo_Hinh": item["Mo_Hinh"],
                "Class": label,
                **values,
            })

test_report_df = pd.DataFrame(test_report_rows)

test_report_df.to_csv(
    OUTPUT_DIR / "Viec14_Classification_Report_Test.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 14. LUU CAU HINH THUC NGHIEM
# ============================================================

config_output = {
    "dataset": str(DATA_PATH),
    "target": TARGET_COL,
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE,
    "cv": "StratifiedKFold",
    "n_splits": N_SPLITS,
    "shuffle": True,
    "scoring": [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ],
    "refit_metric": "recall",
    "feature_scenario": SELECTED_SCENARIO,
    "features": X_train.columns.tolist(),
    "class_weight_comparison": [None, "balanced"],
    "smote_used": False,
    "pipeline_imputer": "median",
    "output_directory": str(OUTPUT_DIR),
}

with open(
    OUTPUT_DIR / "Viec14_Cau_Hinh_Thuc_Nghiem.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        config_output,
        file,
        ensure_ascii=False,
        indent=4,
    )


# ============================================================
# 15. LUU MO HINH TOT NHAT - KHONG TAO THU MUC MODELS
# ============================================================

# Luu truc tiep cac file joblib trong Model/Viec14.
# Xoa khoi block nay neu chi can CSV va PNG.
for model_name, estimator in best_estimators.items():
    safe_name = (
        model_name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )

    joblib.dump(
        {
            "model": estimator,
            "features": X_train.columns.tolist(),
            "scenario": SELECTED_SCENARIO,
            "target": TARGET_COL,
            "model_name": model_name,
        },
        OUTPUT_DIR / f"Viec14_Best_{safe_name}.joblib",
    )


# ============================================================
# 16. KET LUAN TU KET QUA
# ============================================================

best_result = results_df.iloc[0]

print("\n" + "=" * 75)
print("KET QUA TOT NHAT THEO BEST CV RECALL (refit='recall')")
print("=" * 75)

print("Model:", best_result["Mo_Hinh"])
print("Feature scenario:", best_result["Scenario"])
print("Class weight:", best_result["Best_Class_Weight"])
print(f"Best CV Recall:  {best_result['Best_CV_Recall']:.4f}")
print(f"Best CV F1:      {best_result['Best_CV_F1']:.4f}")
print(f"Best CV ROC-AUC: {best_result['Best_CV_ROC_AUC']:.4f}")
print(f"Test Recall:     {best_result['Test_Recall']:.4f}")
print(f"Test F1:         {best_result['Test_F1']:.4f}")
print(f"Test ROC-AUC:    {best_result['Test_ROC_AUC']:.4f}")
print(f"Test Accuracy:   {best_result['Test_Accuracy']:.4f}")

print("\nBang tong hop:")
display(results_df.round(4))

print("\nBang so sanh class_weight:")
display(
    weight_best_df[
        [
            "Mo_Hinh",
            "Class_Weight",
            "CV_Accuracy",
            "CV_Recall",
            "CV_F1",
            "CV_ROC_AUC",
        ]
    ].round(4)
)


# ============================================================
# 17. LIET KE CAC FILE BAN GIAO
# ============================================================

print("\n" + "=" * 75)
print("HOAN THANH VIEC 14")
print("=" * 75)

print("Thu muc ket qua:", OUTPUT_DIR)

print("\nFile PNG:")
for path in sorted(OUTPUT_DIR.glob("*.png")):
    print(" -", path.name)

print("\nFile CSV:")
for path in sorted(OUTPUT_DIR.glob("*.csv")):
    print(" -", path.name)

print("\nFile model:")
for path in sorted(OUTPUT_DIR.glob("*.joblib")):
    print(" -", path.name)
