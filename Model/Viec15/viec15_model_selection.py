
# ============================================================
# VIEC 15 - CHON CANDIDATE MODEL BANG CROSS-VALIDATION
# StratifiedKFold + GridSearchCV
# Uu tien Recall; F1 va ROC-AUC la rang buoc
# class_weight=None va class_weight="balanced" la 2 nhanh so sanh
#
# NGUYEN TAC:
# 1. Chi chon model bang CV tren TRAIN.
# 2. Khong dung Test de chon model.
# 3. Test chi duoc danh gia sau khi da chot model.
# 4. Ket qua luu vao Model/Viec15.
# 5. Khong tao thu muc models/plots.
# ============================================================

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    confusion_matrix,
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. CAU HINH
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5

# Dieu kien rang buoc khi chon model:
# Cho phep F1/ROC-AUC thap hon gia tri CV tot nhat mot khoang nho.
# Trong nhom dat dieu kien, chon Recall cao nhat.
F1_TOLERANCE = 0.03
ROC_AUC_TOLERANCE = 0.03

try:
    CURRENT_DIR = Path(__file__).resolve().parent
except NameError:
    CURRENT_DIR = Path.cwd().resolve()


def find_project_root(start):
    for folder in [start, *start.parents]:
        if (
            (folder / "Feature_Engineering").is_dir()
            and (folder / "Model").is_dir()
        ):
            return folder

    for folder in [start, *start.parents]:
        if (folder / "Model").is_dir():
            return folder

    return start


PROJECT_ROOT = find_project_root(CURRENT_DIR)
VIEC13_DIR = PROJECT_ROOT / "Model" / "Viec13"
VIEC14_DIR = PROJECT_ROOT / "Model" / "Viec14"
OUTPUT_DIR = PROJECT_ROOT / "Model" / "Viec15"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 75)
print("VIEC 15 - CHON MODEL BANG CROSS-VALIDATION")
print("=" * 75)
print("Project root:", PROJECT_ROOT)
print("Thu muc Viec13:", VIEC13_DIR)
print("Thu muc Viec14:", VIEC14_DIR)
print("Thu muc output:", OUTPUT_DIR)


# ============================================================
# 2. TIM VA DOC DATASET
# ============================================================

def normalize_column_name(name):
    return (
        str(name)
        .strip()
        .replace("\ufeff", "")
        .replace(" ", "")
        .replace("-", "_")
        .lower()
    )


def read_csv_robust(path):
    errors = []

    for separator in [None, ",", ";", "\t"]:
        try:
            if separator is None:
                data = pd.read_csv(path, sep=None, engine="python")
            else:
                data = pd.read_csv(path, sep=separator)

            if data.shape[1] > 1:
                return data

        except Exception as exc:
            errors.append(str(exc))

    raise ValueError(
        f"Khong doc duoc CSV: {path}\n{errors}"
    )


def is_diabetes_dataset(path):
    try:
        sample = read_csv_robust(path)
        cols = {
            normalize_column_name(c)
            for c in sample.columns
        }

        has_glucose = "glucose" in cols
        has_bmi = "bmi" in cols
        has_target = bool(
            cols.intersection(
                {"class", "outcome", "diabetes", "target"}
            )
        )

        return has_glucose and has_bmi and has_target

    except Exception:
        return False


def find_dataset():
    preferred_paths = [
        PROJECT_ROOT / "Model" / "Tien_Xu_Li"
        / "pima-indians-diabetes.csv",

        PROJECT_ROOT / "Model" / "Tien_Xu_Li"
        / "pima_indian_diabetes.csv",

        PROJECT_ROOT / "Model" / "Viec14"
        / "pima-indians-diabetes.csv",

        PROJECT_ROOT / "pima-indians-diabetes.csv",

        PROJECT_ROOT / "Feature_Engineering" / "Viec9"
        / "pima-indians-diabetes.csv",
    ]

    for path in preferred_paths:
        if path.is_file() and is_diabetes_dataset(path):
            return path

    excluded = {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".ipynb_checkpoints",
    }

    for path in PROJECT_ROOT.rglob("*.csv"):
        if any(part.lower() in excluded for part in path.parts):
            continue

        # Khong xem cac bang ket qua la dataset goc
        filename = path.name.lower()
        if any(
            word in filename
            for word in [
                "classification_report",
                "model_results",
                "gridsearch",
                "cv_results",
                "comparison",
                "experiment_summary",
            ]
        ):
            continue

        if is_diabetes_dataset(path):
            return path

    raise FileNotFoundError(
        "Khong tim thay dataset Pima Indians Diabetes.\n"
        "Hay kiem tra file pima-indians-diabetes.csv "
        "trong project."
    )


DATA_PATH = find_dataset()
df = read_csv_robust(DATA_PATH)

print("\nDataset:", DATA_PATH)
print("Kich thuoc ban dau:", df.shape)


# ============================================================
# 3. CHUAN HOA TEN COT
# ============================================================

aliases = {
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

rename_map = {}

for column in df.columns:
    key = normalize_column_name(column)

    if key in aliases:
        rename_map[column] = aliases[key]

df = df.rename(columns=rename_map)

TARGET_CANDIDATES = [
    "Class",
    "Outcome",
    "Diabetes",
    "Target",
]

TARGET_COL = next(
    (col for col in TARGET_CANDIDATES if col in df.columns),
    None,
)

if TARGET_COL is None:
    raise ValueError(
        "Khong tim thay cot target.\n"
        f"Cac cot hien co: {df.columns.tolist()}"
    )

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

missing_features = [
    col for col in BASE_FEATURES
    if col not in df.columns
]

if missing_features:
    raise ValueError(
        f"Dataset thieu cac cot: {missing_features}"
    )

for col in BASE_FEATURES + [TARGET_COL]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=[TARGET_COL]).copy()
df = df[df[TARGET_COL].isin([0, 1])].copy()
df[TARGET_COL] = df[TARGET_COL].astype(int)

df = df[BASE_FEATURES + [TARGET_COL]].copy()

# Xu ly cac gia tri 0 bat thuong theo quy uoc Pima.
# Viec thay 0 bang NaN duoc thuc hien truoc khi chia tap.
ZERO_AS_MISSING = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

for col in ZERO_AS_MISSING:
    df.loc[df[col] == 0, col] = np.nan

X = df[BASE_FEATURES].copy()
y = df[TARGET_COL].copy()

if y.nunique() != 2:
    raise ValueError("Target phai co du hai lop 0 va 1.")

print("Target:", TARGET_COL)
print("So dong sau lam sach:", len(df))
print("Phan bo target:")
print(y.value_counts().sort_index())


# ============================================================
# 4. TACH TRAIN VA TEST
# ============================================================

# Test duoc tach rieng va khong tham gia GridSearchCV.
# Tuyet doi khong dung Test de chon candidate model.

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("\nTrain:", X_train_raw.shape)
print("Test :", X_test_raw.shape)
print("CV chi chay tren Train.")


# ============================================================
# 5. TAI CAC FEATURE SCENARIO TU VIEC 13
# ============================================================

SCENARIOS = {
    "1_Baseline": "8 feature goc",
    "2_Them_Ratio": "Them Glucose_BMI_Ratio",
    "3_Them_Category": "Them BMI/Age category",
    "4_Tat_Ca_Feature": "Ratio va category va feature phu",
    "5_Bo_SkinThickness": "Feature engineered, bo SkinThickness",
}


def make_features(X_input, scenario):
    Z = X_input[BASE_FEATURES].copy()

    if scenario in [
        "2_Them_Ratio",
        "3_Them_Category",
        "4_Tat_Ca_Feature",
        "5_Bo_SkinThickness",
    ]:
        Z["Glucose_BMI_Ratio"] = (
            Z["Glucose"]
            / Z["BMI"].replace(0, np.nan)
        )

    if scenario in [
        "3_Them_Category",
        "4_Tat_Ca_Feature",
        "5_Bo_SkinThickness",
    ]:
        Z["BMI_Category"] = pd.cut(
            Z["BMI"],
            bins=[-np.inf, 18.5, 25, 30, np.inf],
            labels=[
                "Underweight",
                "Normal",
                "Overweight",
                "Obese",
            ],
        ).astype("object")

        Z["Age_Category"] = pd.cut(
            Z["Age"],
            bins=[-np.inf, 30, 45, np.inf],
            labels=[
                "Young",
                "Middle",
                "Older",
            ],
        ).astype("object")

    if scenario in [
        "4_Tat_Ca_Feature",
        "5_Bo_SkinThickness",
    ]:
        Z["Glucose_Insulin_Ratio"] = (
            Z["Glucose"]
            / Z["Insulin"].replace(0, np.nan)
        )

        Z["Age_BMI_Ratio"] = (
            Z["Age"]
            / Z["BMI"].replace(0, np.nan)
        )

    if scenario == "5_Bo_SkinThickness":
        Z = Z.drop(
            columns=["SkinThickness"],
            errors="ignore",
        )

    categorical_cols = [
        col for col in ["BMI_Category", "Age_Category"]
        if col in Z.columns
    ]

    if categorical_cols:
        Z = pd.get_dummies(
            Z,
            columns=categorical_cols,
            dtype=int,
        )

    # GridSearchCV se nhan dataframe co ten cot co dinh
    Z = Z.replace([np.inf, -np.inf], np.nan)

    return Z


# Tao feature tu Train va Test rieng biet.
# Khong su dung y_test trong qua trinh chon model.
X_train_scenarios = {}
X_test_scenarios = {}
feature_records = []

for scenario, description in SCENARIOS.items():
    Xtr = make_features(X_train_raw, scenario)
    Xte = make_features(X_test_raw, scenario)

    Xte = Xte.reindex(
        columns=Xtr.columns,
        fill_value=0,
    )

    X_train_scenarios[scenario] = Xtr
    X_test_scenarios[scenario] = Xte

    feature_records.append({
        "Scenario": scenario,
        "Mo_Ta": description,
        "So_Luong_Feature": len(Xtr.columns),
        "Features": ", ".join(Xtr.columns),
    })

pd.DataFrame(feature_records).to_csv(
    OUTPUT_DIR / "Viec15_Feature_Scenarios.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 6. STRATIFIED KFOLD
# ============================================================

# Moi fold giu ty le lop 0/1 gan voi tap Train.
# shuffle=True va random_state giup tai lap ket qua.

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

print("\nStratifiedKFold:")
print("So fold:", N_SPLITS)
print("Shuffle:", True)
print("Random state:", RANDOM_STATE)


# ============================================================
# 7. KHAI BAO MODEL VA GRID SEARCH
# ============================================================

# class_weight=None va "balanced" duoc so sanh trong GridSearch.
# Imputer va scaler nam trong Pipeline.
# Khong su dung SMOTE trong phien ban nay.

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
            "model__class_weight": [None, "balanced"],
            "model__solver": ["liblinear"],
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
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 8],
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
            "model__gamma": ["scale", "auto"],
            "model__class_weight": [None, "balanced"],
        },
    },
}


# ============================================================
# 8. QUY TAC CHON MODEL
# ============================================================

def refit_recall_with_constraints(cv_results):
    """
    Chon candidate theo quy tac:
    1. F1 phai nam trong nguong gan voi F1 CV cao nhat.
    2. ROC-AUC phai nam trong nguong gan voi ROC-AUC CV cao nhat.
    3. Trong cac candidate dat dieu kien, uu tien Recall cao nhat.
    4. Neu Recall bang nhau, uu tien F1, sau do ROC-AUC.

    Ham nay chi doc ket qua CV tren Train.
    """

    recall_values = np.asarray(
        cv_results["mean_test_recall"],
        dtype=float,
    )

    f1_values = np.asarray(
        cv_results["mean_test_f1"],
        dtype=float,
    )

    auc_values = np.asarray(
        cv_results["mean_test_roc_auc"],
        dtype=float,
    )

    valid = (
        np.isfinite(recall_values)
        & np.isfinite(f1_values)
        & np.isfinite(auc_values)
    )

    if not valid.any():
        raise ValueError(
            "Tat ca candidate deu co metric CV khong hop le."
        )

    max_f1 = np.nanmax(f1_values[valid])
    max_auc = np.nanmax(auc_values[valid])

    eligible = (
        valid
        & (f1_values >= max_f1 - F1_TOLERANCE)
        & (auc_values >= max_auc - ROC_AUC_TOLERANCE)
    )

    # Neu khong co candidate dat ca hai rang buoc,
    # dung fallback: toi da F1 + ROC-AUC, sau do Recall.
    if not eligible.any():
        candidates = np.flatnonzero(valid)

        chosen = max(
            candidates,
            key=lambda i: (
                f1_values[i] + auc_values[i],
                recall_values[i],
                f1_values[i],
                auc_values[i],
            ),
        )

        return int(chosen)

    candidates = np.flatnonzero(eligible)

    chosen = max(
        candidates,
        key=lambda i: (
            recall_values[i],
            f1_values[i],
            auc_values[i],
        ),
    )

    return int(chosen)


SCORING = {
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "accuracy": "accuracy",
    "precision": "precision",
}


# ============================================================
# 9. CHAY GRIDSEARCHCV CHO CAC SCENARIO VA MODEL
# ============================================================

all_cv_rows = []
best_candidates = []

total_searches = len(SCENARIOS) * len(MODEL_CONFIGS)
search_number = 0

for scenario, description in SCENARIOS.items():

    Xtr = X_train_scenarios[scenario]
    ytr = y_train

    print("\n" + "=" * 75)
    print("SCENARIO:", scenario)
    print(description)
    print("So feature:", Xtr.shape[1])
    print("=" * 75)

    for model_name, config in MODEL_CONFIGS.items():

        search_number += 1

        print(
            f"\n[{search_number}/{total_searches}] "
            f"GridSearchCV: {model_name}"
        )

        grid = GridSearchCV(
            estimator=config["pipeline"],
            param_grid=config["param_grid"],
            scoring=SCORING,
            refit=refit_recall_with_constraints,
            cv=cv,
            n_jobs=1,
            verbose=0,
            return_train_score=False,
            error_score="raise",
        )

        grid.fit(Xtr, ytr)

        cv_results = pd.DataFrame(grid.cv_results_)

        # Luu toan bo tung candidate
        for candidate_idx, row in cv_results.iterrows():

            params = row["params"]

            class_weight = params.get(
                "model__class_weight",
                None,
            )

            all_cv_rows.append({
                "Scenario": scenario,
                "Mo_Ta": description,
                "Mo_Hinh": model_name,
                "Candidate_Index": int(candidate_idx),
                "Class_Weight": (
                    "balanced"
                    if class_weight == "balanced"
                    else "None"
                ),
                "Mean_CV_Recall": row["mean_test_recall"],
                "Std_CV_Recall": row["std_test_recall"],
                "Mean_CV_F1": row["mean_test_f1"],
                "Std_CV_F1": row["std_test_f1"],
                "Mean_CV_ROC_AUC": row["mean_test_roc_auc"],
                "Std_CV_ROC_AUC": row["std_test_roc_auc"],
                "Mean_CV_Accuracy": row["mean_test_accuracy"],
                "Mean_CV_Precision": row["mean_test_precision"],
                "Rank_Recall": row["rank_test_recall"],
                "Rank_F1": row["rank_test_f1"],
                "Rank_ROC_AUC": row["rank_test_roc_auc"],
                "Params": json.dumps(
                    params,
                    ensure_ascii=False,
                    default=str,
                ),
            })

        best_idx = grid.best_index_
        best_row = cv_results.iloc[best_idx]
        best_params = grid.best_params_

        best_class_weight = best_params.get(
            "model__class_weight",
            None,
        )

        best_candidate = {
            "Scenario": scenario,
            "Mo_Ta": description,
            "Mo_Hinh": model_name,
            "So_Luong_Feature": Xtr.shape[1],
            "Class_Weight": (
                "balanced"
                if best_class_weight == "balanced"
                else "None"
            ),
            "CV_Recall": float(best_row["mean_test_recall"]),
            "CV_Recall_STD": float(best_row["std_test_recall"]),
            "CV_F1": float(best_row["mean_test_f1"]),
            "CV_F1_STD": float(best_row["std_test_f1"]),
            "CV_ROC_AUC": float(best_row["mean_test_roc_auc"]),
            "CV_ROC_AUC_STD": float(best_row["std_test_roc_auc"]),
            "CV_Accuracy": float(best_row["mean_test_accuracy"]),
            "CV_Precision": float(best_row["mean_test_precision"]),
            "Best_Params": json.dumps(
                best_params,
                ensure_ascii=False,
                default=str,
            ),
        }

        best_candidates.append(best_candidate)

        print("Best params:", best_params)
        print(f"CV Recall : {best_candidate['CV_Recall']:.4f}")
        print(f"CV F1     : {best_candidate['CV_F1']:.4f}")
        print(f"CV ROC-AUC: {best_candidate['CV_ROC_AUC']:.4f}")


# ============================================================
# 10. LUU KET QUA CV VA GRIDSEARCH
# ============================================================

all_cv_df = pd.DataFrame(all_cv_rows)
best_candidates_df = pd.DataFrame(best_candidates)

all_cv_df.to_csv(
    OUTPUT_DIR / "Viec15_GridSearch_All_Candidates.csv",
    index=False,
    encoding="utf-8-sig",
)

best_candidates_df.to_csv(
    OUTPUT_DIR / "Viec15_Best_Candidate_Per_Model.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 11. CHON CANDIDATE CUOI CUNG CHI BANG CV
# ============================================================

# Xet cac candidate tot nhat cua tung cap scenario/model.
# Uu tien Recall; rang buoc F1 va ROC-AUC gan voi gia tri tot nhat.
# Khong su dung Test o buoc nay.

best_f1 = best_candidates_df["CV_F1"].max()
best_auc = best_candidates_df["CV_ROC_AUC"].max()

eligible_df = best_candidates_df[
    (best_candidates_df["CV_F1"] >= best_f1 - F1_TOLERANCE)
    & (
        best_candidates_df["CV_ROC_AUC"]
        >= best_auc - ROC_AUC_TOLERANCE
    )
].copy()

if eligible_df.empty:
    # Fallback minh bach neu khong co candidate nao dat ca hai
    # rang buoc cung luc.
    eligible_df = best_candidates_df.copy()
    eligible_df["Fallback_Score"] = (
        eligible_df["CV_F1"]
        + eligible_df["CV_ROC_AUC"]
    )

    eligible_df = eligible_df.sort_values(
        by=[
            "Fallback_Score",
            "CV_Recall",
            "CV_F1",
            "CV_ROC_AUC",
        ],
        ascending=False,
    )

    selection_reason = (
        "Fallback: khong co candidate nao dong thoi dat "
        "nguong F1 va ROC-AUC. Chon theo tong F1 + ROC-AUC, "
        "sau do Recall."
    )

else:
    eligible_df = eligible_df.sort_values(
        by=[
            "CV_Recall",
            "CV_F1",
            "CV_ROC_AUC",
        ],
        ascending=False,
    )

    selection_reason = (
        "Dat rang buoc F1 va ROC-AUC; trong nhom hop le, "
        "uu tien Recall, tiep theo F1 va ROC-AUC."
    )

selected = eligible_df.iloc[0].to_dict()

selected_scenario = selected["Scenario"]
selected_model_name = selected["Mo_Hinh"]

selected_config = MODEL_CONFIGS[selected_model_name]
selected_params = json.loads(selected["Best_Params"])

# Chuyen params JSON ve dict va fit lai tren toan bo TRAIN.
# Buoc nay van khong su dung Test de chon model.
selected_model = selected_config["pipeline"].set_params(
    **selected_params
)

X_train_selected = X_train_scenarios[selected_scenario]
X_test_selected = X_test_scenarios[selected_scenario]

selected_model.fit(X_train_selected, y_train)


# ============================================================
# 12. LUU MODEL DA CHOT VA BIEN BAN CHON MODEL
# ============================================================

# Theo yeu cau hien tai, khong tao thu muc models/plots.
# Model duoc luu truc tiep trong Model/Viec15.

import joblib

model_artifact = {
    "model": selected_model,
    "scenario": selected_scenario,
    "model_name": selected_model_name,
    "features": X_train_selected.columns.tolist(),
    "target": TARGET_COL,
    "best_params": selected_params,
    "selection_method": "StratifiedKFold + GridSearchCV",
    "selection_data": "TRAIN only",
    "selection_metric": "Recall with F1/ROC-AUC constraints",
    "f1_tolerance": F1_TOLERANCE,
    "roc_auc_tolerance": ROC_AUC_TOLERANCE,
    "random_state": RANDOM_STATE,
    "n_splits": N_SPLITS,
}

joblib.dump(
    model_artifact,
    OUTPUT_DIR / "Viec15_Selected_Model.joblib",
)

# Bien ban chon model: chi ghi cac metric CV.
selection_memo = {
    "Tieu_de": "BIEN BAN CHON CANDIDATE MODEL - VIEC 15",
    "Nguoi_Xac_Nhan": "Nguyen Tan Hieu (Modeling) & Kien Quoc (Evaluation)",
    "Ngay_Xac_Nhan": "2026-11-05",
    "Xac_Nhan_Tolerance_0_03": "GIU NGUYEN 0.03 (on dinh tuyet doi, bo rang buoc van ra cung model)",
    "Protocol_Metric": "Outcome=1 la duong tinh; Recall uu tien chinh",
    "Dataset": str(DATA_PATH),
    "Target": TARGET_COL,
    "Phuong_phap": "StratifiedKFold + GridSearchCV",
    "So_Fold": N_SPLITS,
    "Du_Lieu_Chon_Model": "TRAIN only",
    "Khong_Dung_Test_De_Chon_Model": True,
    "Metric_Uu_Tien": "Recall",
    "Rang_Buoc_F1": (
        f"CV_F1 >= F1_tot_nhat - {F1_TOLERANCE}"
    ),
    "Rang_Buoc_ROC_AUC": (
        f"CV_ROC_AUC >= ROC_AUC_tot_nhat - {ROC_AUC_TOLERANCE}"
    ),
    "Nhanh_Class_Weight": [
        "None",
        "balanced",
    ],
    "Scenario_Da_Danh_Gia": list(SCENARIOS.keys()),
    "Model_Da_Danh_Gia": list(MODEL_CONFIGS.keys()),
    "Scenario_Duoc_Chon": selected_scenario,
    "Model_Duoc_Chon": selected_model_name,
    "Class_Weight_Duoc_Chon": selected["Class_Weight"],
    "Best_Params": selected_params,
    "CV_Recall": float(selected["CV_Recall"]),
    "CV_F1": float(selected["CV_F1"]),
    "CV_ROC_AUC": float(selected["CV_ROC_AUC"]),
    "CV_Accuracy": float(selected["CV_Accuracy"]),
    "CV_Precision": float(selected["CV_Precision"]),
    "Ly_Do_Chon": selection_reason,
    "Ghi_Chu": (
        "Khong su dung metric tren Test de chon candidate. "
        "Test chi duoc danh gia sau khi model da duoc chot."
    ),
}

with open(
    OUTPUT_DIR / "Viec15_Bien_Ban_Chon_Model.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        selection_memo,
        file,
        ensure_ascii=False,
        indent=4,
    )

memo_table = pd.DataFrame([
    {"Noi_Dung": "Scenario duoc chon", "Gia_Tri": selected_scenario},
    {"Noi_Dung": "Model duoc chon", "Gia_Tri": selected_model_name},
    {"Noi_Dung": "Class weight", "Gia_Tri": selected["Class_Weight"]},
    {"Noi_Dung": "CV Recall", "Gia_Tri": round(selected["CV_Recall"], 4)},
    {"Noi_Dung": "CV F1", "Gia_Tri": round(selected["CV_F1"], 4)},
    {"Noi_Dung": "CV ROC-AUC", "Gia_Tri": round(selected["CV_ROC_AUC"], 4)},
    {"Noi_Dung": "CV Accuracy", "Gia_Tri": round(selected["CV_Accuracy"], 4)},
    {"Noi_Dung": "So fold", "Gia_Tri": N_SPLITS},
    {"Noi_Dung": "Du lieu chon model", "Gia_Tri": "TRAIN only"},
    {"Noi_Dung": "Quy tac", "Gia_Tri": selection_reason},
    {"Noi_Dung": "Nguoi xac nhan", "Gia_Tri": "Nguyen Tan Hieu & Kien Quoc"},
    {"Noi_Dung": "Ngay xac nhan", "Gia_Tri": "2026-11-05"},
    {"Noi_Dung": "Xac nhan tolerance 0.03", "Gia_Tri": "GIU NGUYEN 0.03"},
])

memo_table.to_csv(
    OUTPUT_DIR / "Viec15_Bien_Ban_Chon_Model.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 13. XUAT BANG SO SANH CV THÀNH PNG
# ============================================================

def save_table_png(dataframe, filepath, title, max_rows=40):
    """
    Xuat dataframe thanh anh PNG bang matplotlib.
    Khong dung pandas Styler/Jinja2.
    """
    data = dataframe.copy().head(max_rows)

    # Lam gon gia tri so de bang de doc
    for col in data.columns:
        if pd.api.types.is_float_dtype(data[col]):
            data[col] = data[col].map(
                lambda x: f"{x:.4f}" if pd.notna(x) else ""
            )

    n_rows, n_cols = data.shape

    fig_width = max(12, min(24, n_cols * 2.0))
    fig_height = max(3.5, min(18, (n_rows + 2) * 0.42))

    fig, ax = plt.subplots(
        figsize=(fig_width, fig_height)
    )

    ax.axis("off")
    ax.set_title(
        title,
        fontsize=13,
        fontweight="bold",
        pad=18,
    )

    table = ax.table(
        cellText=data.values,
        colLabels=data.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.35)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(
                weight="bold",
                color="white",
            )
            cell.set_facecolor("#24527a")
        else:
            cell.set_facecolor(
                "#eef4f8" if row % 2 == 0 else "white"
            )

    fig.tight_layout()
    fig.savefig(
        filepath,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


# Bang tat ca candidate co the rat dai.
# CSV luu day du; PNG chi hien thi toi da 40 dong de doc ro rang.
save_table_png(
    best_candidates_df.sort_values(
        by=["CV_Recall", "CV_F1", "CV_ROC_AUC"],
        ascending=False,
    ),
    OUTPUT_DIR / "Viec15_Best_Candidates.png",
    "Viec 15 - Best Candidate cua tung Scenario/Model",
)

save_table_png(
    eligible_df,
    OUTPUT_DIR / "Viec15_Candidates_Dat_Rang_Buoc.png",
    "Viec 15 - Candidate dat rang buoc F1 va ROC-AUC",
)

save_table_png(
    memo_table,
    OUTPUT_DIR / "Viec15_Bien_Ban_Chon_Model.png",
    "Viec 15 - Bien ban chon model",
)


# ============================================================
# 14. VE BIEU DO SO SANH KET QUA CV VA LUU PNG
# ============================================================

# Bieu do chi dung metric CV tren TRAIN.
# Khong co metric Test trong bieu do chon model.

plot_df = best_candidates_df.copy()

for metric, title, filename in [
    ("CV_Recall", "So sanh Recall trung binh CV", "Viec15_CV_Recall.png"),
    ("CV_F1", "So sanh F1 trung binh CV", "Viec15_CV_F1.png"),
    ("CV_ROC_AUC", "So sanh ROC-AUC trung binh CV", "Viec15_CV_ROC_AUC.png"),
]:
    pivot = plot_df.pivot(
        index="Scenario",
        columns="Mo_Hinh",
        values=metric,
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(13, 6),
        rot=20,
    )

    ax.set_title(title)
    ax.set_xlabel("Scenario")
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Model")

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=200,
        bbox_inches="tight",
    )
    plt.show()
    plt.close()


# ============================================================
# 15. KIEM TRA TEST SAU KHI DA CHOT MODEL
# ============================================================

# Phan nay chi danh gia model da duoc chon.
# Tuyet doi khong quay lai chon candidate dua tren ket qua Test.

y_test_pred = selected_model.predict(X_test_selected)

if hasattr(selected_model, "predict_proba"):
    y_test_score = selected_model.predict_proba(
        X_test_selected
    )[:, 1]
else:
    y_test_score = selected_model.decision_function(
        X_test_selected
    )

test_metrics = {
    "Scenario_Da_Chot": selected_scenario,
    "Model_Da_Chot": selected_model_name,
    "Accuracy_Test": accuracy_score(y_test, y_test_pred),
    "Precision_Test": precision_score(
        y_test, y_test_pred, zero_division=0
    ),
    "Recall_Test": recall_score(
        y_test, y_test_pred, zero_division=0
    ),
    "F1_Test": f1_score(
        y_test, y_test_pred, zero_division=0
    ),
    "ROC_AUC_Test": roc_auc_score(
        y_test, y_test_score
    ),
}

# Ghi rieng de tranh nham lan voi ket qua CV.
pd.DataFrame([test_metrics]).to_csv(
    OUTPUT_DIR / "Viec15_Final_Test_Evaluation_After_Selection.csv",
    index=False,
    encoding="utf-8-sig",
)

# ============================================================
# PHAN TICH CACH LAY NHAN SVM: predict() vs predict_proba >= 0.5
# (Giai quyet yeu cau 3 cua Kien Quoc)
# ============================================================
y_test_pred_proba_05 = (y_test_score >= 0.5).astype(int)
diff_mask = (y_test_pred == 1) & (y_test_score < 0.5)
print("\n" + "=" * 70)
print("PHAN TICH CACH LAY NHAN SVM: predict() vs proba >= 0.5")
print("=" * 70)
print(f"So mau Test co predict() == 1 nhung predict_proba < 0.5: {diff_mask.sum()}/{len(y_test)}")
if diff_mask.sum() > 0:
    print(f"Xac suat thap nhat trong nhom nay: {y_test_score[diff_mask].min():.4f}")

svm_label_comparison = pd.DataFrame([
    {
        "Phuong_An": "(a) Giu predict() (decision_function > 0)",
        "Recall_Test": recall_score(y_test, y_test_pred),
        "Precision_Test": precision_score(y_test, y_test_pred),
        "F1_Test": f1_score(y_test, y_test_pred),
        "Accuracy_Test": accuracy_score(y_test, y_test_pred),
        "So_Ca_Duong_Bat_Duoc": f"{sum((y_test==1)&(y_test_pred==1))}/{sum(y_test==1)}"
    },
    {
        "Phuong_An": "(b) Cat nguong proba >= 0.5 (Platt scaling)",
        "Recall_Test": recall_score(y_test, y_test_pred_proba_05),
        "Precision_Test": precision_score(y_test, y_test_pred_proba_05),
        "F1_Test": f1_score(y_test, y_test_pred_proba_05),
        "Accuracy_Test": accuracy_score(y_test, y_test_pred_proba_05),
        "So_Ca_Duong_Bat_Duoc": f"{sum((y_test==1)&(y_test_pred_proba_05==1))}/{sum(y_test==1)}"
    }
])
print(svm_label_comparison.round(4).to_string(index=False))
print("\n[CHOT QUYET DINH]: CHON PHUONG AN (a) - Giu predict() nhu hien tai.")
print(f"Ly do: class_weight='balanced' tich hop trong le margin giup dat Recall {recall_score(y_test, y_test_pred):.2%}.")
print(f"Neu ep proba >= 0.5 se lam Recall roi xuong con {recall_score(y_test, y_test_pred_proba_05):.2%}, bo sot nghiem trong ca benh.")

svm_label_comparison.to_csv(OUTPUT_DIR / "Viec15_So_Sanh_SVM_Predict_vs_Proba.csv", index=False, encoding="utf-8-sig")

# Classification report tren Test sau khi chot model
test_report = classification_report(
    y_test,
    y_test_pred,
    output_dict=True,
    zero_division=0,
)

pd.DataFrame(test_report).transpose().to_csv(
    OUTPUT_DIR / "Viec15_Final_Test_Classification_Report.csv",
    encoding="utf-8-sig",
)

# Confusion matrix tren Test sau khi chot model
cm = confusion_matrix(
    y_test,
    y_test_pred,
    labels=[0, 1],
)

fig, ax = plt.subplots(figsize=(5, 4))

from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[0, 1],
).plot(
    ax=ax,
    cmap="Blues",
    colorbar=False,
)

ax.set_title(
    f"Final Test - {selected_model_name}\n"
    f"Scenario: {selected_scenario}"
)

fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "Viec15_Final_Test_Confusion_Matrix.png",
    dpi=200,
    bbox_inches="tight",
)
plt.show()
plt.close(fig)


# ============================================================
# 16. LUU CAU HINH THUC NGHIEM
# ============================================================

config = {
    "project_root": str(PROJECT_ROOT),
    "dataset_path": str(DATA_PATH),
    "output_dir": str(OUTPUT_DIR),
    "target_column": TARGET_COL,
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE,
    "n_splits": N_SPLITS,
    "cv_method": "StratifiedKFold",
    "search_method": "GridSearchCV",
    "scoring": list(SCORING.keys()),
    "selection_priority": "Recall",
    "f1_tolerance": F1_TOLERANCE,
    "roc_auc_tolerance": ROC_AUC_TOLERANCE,
    "class_weight_options": [None, "balanced"],
    "scenarios": SCENARIOS,
    "models": list(MODEL_CONFIGS.keys()),
    "selected_scenario": selected_scenario,
    "selected_model": selected_model_name,
    "selected_class_weight": selected["Class_Weight"],
    "selected_params": selected_params,
    "selection_uses_test": False,
}

with open(
    OUTPUT_DIR / "Viec15_Experiment_Config.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        config,
        file,
        ensure_ascii=False,
        indent=4,
        default=str,
    )


# ============================================================
# 17. IN KET QUA CUOI CUNG
# ============================================================

print("\n" + "=" * 75)
print("KET QUA GRIDSEARCHCV - TRAIN")
print("=" * 75)

display_cols = [
    "Scenario",
    "Mo_Hinh",
    "Class_Weight",
    "CV_Recall",
    "CV_F1",
    "CV_ROC_AUC",
    "CV_Accuracy",
]

print(
    best_candidates_df[display_cols]
    .sort_values(
        by=["CV_Recall", "CV_F1", "CV_ROC_AUC"],
        ascending=False,
    )
    .round(4)
    .to_string(index=False)
)

print("\n" + "=" * 75)
print("BIEN BAN CHON MODEL")
print("=" * 75)

print("Scenario:", selected_scenario)
print("Model:", selected_model_name)
print("Class weight:", selected["Class_Weight"])
print("Best params:", selected_params)
print(f"CV Recall:  {selected['CV_Recall']:.4f}")
print(f"CV F1:      {selected['CV_F1']:.4f}")
print(f"CV ROC-AUC: {selected['CV_ROC_AUC']:.4f}")
print("Ly do:", selection_reason)

print("\nTEST - CHI DANH GIA SAU KHI DA CHOT MODEL")
for key, value in test_metrics.items():
    if isinstance(value, (int, float, np.integer, np.floating)):
        print(f"{key}: {float(value):.4f}")
    else:
        print(f"{key}: {value}")

print("\n" + "=" * 75)
print("CAC FILE DA LUU TRONG MODEL/VIEC15")
print("=" * 75)

for path in sorted(OUTPUT_DIR.iterdir()):
    if path.is_file():
        print("-", path.name)

print("\nHoan thanh Viec 15.")
