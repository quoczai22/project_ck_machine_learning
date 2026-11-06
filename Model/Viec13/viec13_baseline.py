
# ============================================================
# VIEC 13 - BASELINE MODEL EVALUATION
# Logistic Regression | Random Forest | SVM
# Dataset: Pima Indians Diabetes
# Chay truc tiep trong EDA.ipynb - khong dung __file__
# ============================================================

import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from IPython.display import display

from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.dummy import DummyClassifier
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
    average_precision_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. CAU HINH DU AN VA THU MUC LUU
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

try:
    CURRENT_DIR = Path(__file__).resolve().parent
except NameError:
    CURRENT_DIR = Path.cwd().resolve()


def find_project_root(start):
    """
    Tim thu muc goc co Feature_Engineering hoac Model.
    Ho tro notebook dat trong project hoac cac thu muc con.
    """
    for p in [start, *start.parents]:
        if (
            (p / "Feature_Engineering").is_dir()
            or (p / "Model").is_dir()
        ):
            return p

    return start


PROJECT_ROOT = find_project_root(CURRENT_DIR)

# Thu muc output uu tien theo cau truc project trong anh
OUTPUT_CANDIDATES = [
    PROJECT_ROOT / "Model" / "Tien_Xu_Li" / "Viec13",
    PROJECT_ROOT / "Model" / "Viec13",
    PROJECT_ROOT / "Tien_Xu_Li" / "Viec13",
]

# Neu Viec13 da ton tai thi dung no.
# Neu chua co, tao tai vi tri dau tien.
OUTPUT_DIR = next(
    (p for p in OUTPUT_CANDIDATES if p.is_dir()),
    OUTPUT_CANDIDATES[0]
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = OUTPUT_DIR / "models"
PLOT_DIR = OUTPUT_DIR / "plots"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("VIEC 13 - BASELINE MODEL EVALUATION")
print("=" * 70)
print("Notebook directory:", CURRENT_DIR)
print("Project root:", PROJECT_ROOT)
print("Output directory:", OUTPUT_DIR)


# ============================================================
# 2. TIM FILE DATASET
# ============================================================

DATASET_NAMES = [
    "pima-indians-diabetes.csv",
    "diabetes.csv",
    "pima.csv",
]


def read_csv_robust(path):
    """
    Thu doc CSV bang cach tu phat hien separator.
    Neu khong thanh cong thi thu cac separator pho bien.
    """
    errors = []

    for kwargs in [
        {"sep": None, "engine": "python"},
        {"sep": ","},
        {"sep": ";"},
        {"sep": "\t"},
    ]:
        try:
            data = pd.read_csv(path, **kwargs)

            # Neu header bi gom thanh mot cot co dau phay
            if data.shape[1] == 1:
                first_col = str(data.columns[0])

                if "," in first_col:
                    retry = pd.read_csv(path, sep=",")
                    if retry.shape[1] > 1:
                        return retry

                if ";" in first_col:
                    retry = pd.read_csv(path, sep=";")
                    if retry.shape[1] > 1:
                        return retry

            if data.shape[1] > 1:
                return data

        except Exception as exc:
            errors.append(str(exc))

    raise ValueError(
        f"Khong doc duoc file CSV: {path}\n{errors}"
    )


def normalize_column_name(name):
    return (
        str(name)
        .strip()
        .replace("\ufeff", "")
        .replace(" ", "")
        .replace("-", "_")
        .lower()
    )


def find_dataset():
    """
    Tim dataset Pima trong project.
    Khong su dung file tong hop ket qua cua Viec9.
    """

    # Uu tien cac vi tri thuong gap
    preferred = [
        PROJECT_ROOT / "Model" / "Tien_Xu_Li"
        / "pima-indians-diabetes.csv",

        PROJECT_ROOT / "Model"
        / "pima-indians-diabetes.csv",

        PROJECT_ROOT / "Model" / "Tien_Xu_Li"
        / "Viec6" / "pima-indians-diabetes.csv",

        PROJECT_ROOT / "Feature_Engineering"
        / "Viec9" / "pima-indians-diabetes.csv",

        PROJECT_ROOT / "Feature_Engineering"
        / "Viec9" / "bang_baseline_feature_set.csv",
    ]

    for path in preferred:
        if path.is_file():
            try:
                sample = read_csv_robust(path)
                cols = {
                    normalize_column_name(c)
                    for c in sample.columns
                }

                target_exists = bool(
                    cols.intersection(
                        {"class", "outcome", "diabetes", "target"}
                    )
                )

                if target_exists and len(cols) >= 5:
                    return path
            except Exception:
                continue

    excluded_dirs = {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        "models",
        "plots",
        ".ipynb_checkpoints",
    }

    # Tim trong project
    for root in [PROJECT_ROOT, CURRENT_DIR]:
        if not root.exists():
            continue

        for path in root.rglob("*.csv"):

            if any(
                part.lower() in excluded_dirs
                for part in path.parts
            ):
                continue

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

                if has_glucose and has_bmi and has_target:
                    return path

            except Exception:
                continue

    raise FileNotFoundError(
        "Khong tim thay dataset Pima Indians Diabetes.\n"
        "Hay kiem tra file CSV trong project, "
        "dac biet la pima-indians-diabetes.csv."
    )


DATA_PATH = find_dataset()

df = read_csv_robust(DATA_PATH)

# Chuan hoa ten cot
df.columns = [
    str(c).strip().replace("\ufeff", "")
    for c in df.columns
]

print("\nDataset duoc su dung:", DATA_PATH)
print("Kich thuoc ban dau:", df.shape)
print("Cac cot ban dau:", df.columns.tolist())

# Neu van con mot cot gom tat ca ten truong
if df.shape[1] == 1 and "," in str(df.columns[0]):
    df = pd.read_csv(DATA_PATH, sep=",")
    df.columns = [
        str(c).strip().replace("\ufeff", "")
        for c in df.columns
    ]

print("Kich thuoc sau khi doc:", df.shape)
print("Cac cot sau khi doc:", df.columns.tolist())

if df.shape[1] < 2:
    raise ValueError(
        "CSV van chi co mot cot. "
        "Hay kiem tra delimiter va noi dung file."
    )


# ============================================================
# 3. CHUAN HOA COT VA XAC DINH TARGET
# ============================================================

rename_map = {}

for col in df.columns:
    normalized = normalize_column_name(col)

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

    if normalized in aliases:
        rename_map[col] = aliases[normalized]

df = df.rename(columns=rename_map)

TARGET_CANDIDATES = [
    "Class",
    "Outcome",
    "Diabetes",
    "Target",
]

TARGET_COL = next(
    (c for c in TARGET_CANDIDATES if c in df.columns),
    None
)

if TARGET_COL is None:
    raise ValueError(
        "Khong tim thay cot nhan Class/Outcome/Diabetes/Target.\n"
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

missing_cols = [
    col for col in BASE_FEATURES
    if col not in df.columns
]

if missing_cols:
    raise ValueError(
        f"Dataset thieu cac cot: {missing_cols}\n"
        f"Cac cot hien co: {df.columns.tolist()}"
    )

# Chuyen du lieu thanh numeric
for col in BASE_FEATURES + [TARGET_COL]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Bo dong khong co nhan
df = df.dropna(subset=[TARGET_COL]).copy()

# Nhan nhi phan 0 va 1
df = df[df[TARGET_COL].isin([0, 1])].copy()
df[TARGET_COL] = df[TARGET_COL].astype(int)

# Chi su dung cac cot can thiet
df = df[BASE_FEATURES + [TARGET_COL]].copy()

print("\nCot target:", TARGET_COL)
print("Kich thuoc dataset:", df.shape)
print("\nPhan bo target:")
print(df[TARGET_COL].value_counts().sort_index())


# ============================================================
# 4. XU LY CAC GIA TRI 0 BAT THUONG
# ============================================================

# Theo quy uoc bo du lieu Pima:
# 0 o cac cot nay duoc xem la gia tri thieu bat thuong.
# Pregnancies = 0 van la gia tri hop le.

ZERO_AS_MISSING = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

for col in ZERO_AS_MISSING:
    df.loc[df[col] == 0, col] = np.nan

# Luu thong tin dataset
df.describe().to_csv(
    OUTPUT_DIR / "dataset_summary.csv",
    encoding="utf-8-sig"
)

df.isna().sum().rename("Missing_Count").to_csv(
    OUTPUT_DIR / "missing_values.csv",
    encoding="utf-8-sig"
)

print("\nSo gia tri thieu theo cot:")
print(df.isna().sum())


# ============================================================
# 5. CHIA TRAIN VA TEST
# ============================================================

X = df[BASE_FEATURES].copy()
y = df[TARGET_COL].copy()

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nSo mau train:", len(X_train_raw))
print("So mau test :", len(X_test_raw))


# ============================================================
# 6. FEATURE ENGINEERING - VIEC 9
# ============================================================

SCENARIOS = {
    "1_Baseline": "8 feature goc",
    "2_Them_Ratio": "Them Glucose_BMI_Ratio",
    "3_Them_Category": "Them Ratio va Category BMI/Age",
    "4_Tat_Ca_Feature": "Them Ratio va Category va feature phu",
    "5_Bo_SkinThickness": "Feature engineered, bo SkinThickness",
}


def make_features(X_input, scenario):
    """
    Tao feature tu du lieu goc.
    Khong su dung target trong feature engineering.
    """

    Z = X_input[BASE_FEATURES].copy()

    # Ratio co xu ly chia cho 0
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

    # Category BMI va Age
    if scenario in [
        "3_Them_Category",
        "4_Tat_Ca_Feature",
        "5_Bo_SkinThickness",
    ]:
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
            labels=[
                "Young",
                "Middle",
                "Older",
            ],
        )

        Z["BMI_Category"] = bmi_category.astype("object")
        Z["Age_Category"] = age_category.astype("object")

    # Them feature phu cho kich ban 4 va 5
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

    # Loai bo SkinThickness neu duoc yeu cau
    if scenario == "5_Bo_SkinThickness":
        Z = Z.drop(
            columns=["SkinThickness"],
            errors="ignore",
        )

    # Chuyen category thanh one-hot
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

    return Z


# ============================================================
# 7. DINH NGHIA CAC MO HINH
# ============================================================

MODELS = {
    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
        )),
    ]),

    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ]),

    "SVM": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", SVC(
            kernel="rbf",
            probability=True,
            random_state=RANDOM_STATE,
        )),
    ]),
}



# ============================================================
# 7.1. DANH GIA BASELINE MODEL BANG 5-FOLD CV TREN TRAIN (CONG 5)
# Giai quyet yeu cau nghiem thu Viec 13 & 17 cua Kien Quoc:
# - DummyClassifier (most_frequent & stratified)
# - Baseline LR, RF, SVM chua tuning bang 5-fold CV tren Train
# - Xac nhan Protocol Metric: Outcome=1 la duong tinh, Recall uu tien chinh
# ============================================================

print("\n" + "=" * 70)
print("DANH GIA BASELINE MODEL BANG 5-FOLD CV TREN TRAIN (CONG 5)")
print("=" * 70)

BASELINE_MODELS = {
    "Dummy (Most Frequent)": Pipeline([
        ("model", DummyClassifier(strategy="most_frequent"))
    ]),
    "Dummy (Stratified)": Pipeline([
        ("model", DummyClassifier(strategy="stratified", random_state=RANDOM_STATE))
    ]),
    "Logistic Regression (Default)": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(random_state=RANDOM_STATE, max_iter=2000))
    ]),
    "Random Forest (Default)": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(random_state=RANDOM_STATE))
    ]),
    "SVM (Default)": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE))
    ]),
}

cv_baseline = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
baseline_cv_rows = []

for b_name, b_pipe in BASELINE_MODELS.items():
    f_recalls, f_precisions, f_f1s, f_aucs, f_praucs, f_accs = [], [], [], [], [], []
    
    for fold, (tr_idx, val_idx) in enumerate(cv_baseline.split(X_train_raw, y_train), 1):
        X_tr, X_val = X_train_raw.iloc[tr_idx], X_train_raw.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        
        pipe_fold = clone(b_pipe)
        pipe_fold.fit(X_tr, y_tr)
        
        y_p = pipe_fold.predict(X_val)
        if hasattr(pipe_fold, "predict_proba"):
            y_s = pipe_fold.predict_proba(X_val)[:, 1]
        elif hasattr(pipe_fold, "decision_function"):
            y_s = pipe_fold.decision_function(X_val)
        else:
            y_s = y_p
            
        f_recalls.append(recall_score(y_val, y_p, zero_division=0))
        f_precisions.append(precision_score(y_val, y_p, zero_division=0))
        f_f1s.append(f1_score(y_val, y_p, zero_division=0))
        f_accs.append(accuracy_score(y_val, y_p))
        try:
            f_aucs.append(roc_auc_score(y_val, y_s))
        except Exception:
            f_aucs.append(0.5)
        try:
            f_praucs.append(average_precision_score(y_val, y_s))
        except Exception:
            f_praucs.append(y_val.mean())
            
    baseline_cv_rows.append({
        "Mo_Hinh": b_name,
        "CV_Recall_Mean": np.mean(f_recalls),
        "CV_Recall_Std": np.std(f_recalls),
        "CV_F1_Mean": np.mean(f_f1s),
        "CV_F1_Std": np.std(f_f1s),
        "CV_ROC_AUC_Mean": np.mean(f_aucs),
        "CV_ROC_AUC_Std": np.std(f_aucs),
        "CV_PR_AUC_Mean": np.mean(f_praucs),
        "CV_Precision_Mean": np.mean(f_precisions),
        "CV_Accuracy_Mean": np.mean(f_accs),
    })

baseline_cv_df = pd.DataFrame(baseline_cv_rows)
# Luu ket qua baseline phuc vu Cong 5 va repo:
baseline_cv_df.to_csv(OUTPUT_DIR / "baseline_cv_train_summary.csv", index=False, encoding="utf-8-sig")
baseline_cv_df.to_csv(OUTPUT_DIR / "baseline_model_results.csv", index=False, encoding="utf-8-sig")

print("\nBang ket qua Baseline Model (5-fold CV tren Train):")
print(baseline_cv_df[["Mo_Hinh", "CV_Recall_Mean", "CV_F1_Mean", "CV_ROC_AUC_Mean", "CV_Accuracy_Mean"]].round(4).to_string(index=False))

# Xac nhan Protocol Metric (PCCV Viec 17)
protocol_metric_info = {
    "Positive_Class": "Outcome = 1",
    "Primary_Metric": "Recall",
    "Secondary_Metrics": ["Precision", "F1", "ROC-AUC", "PR-AUC", "Accuracy"],
    "Xac_Nhan_Boi": "Nguyen Tan Hieu (Modeling) & Kien Quoc (Evaluation)",
    "Ngay_Xac_Nhan": "2026-11-05",
    "Trang_Thai": "DA CHOT"
}
with open(OUTPUT_DIR / "protocol_metric_confirmation.json", "w", encoding="utf-8") as f:
    json.dump(protocol_metric_info, f, ensure_ascii=False, indent=4)
print("\n[OK] Da chot Protocol Metric: Outcome=1 la duong tinh; Recall uu tien hang dau.")
# ============================================================
# 8. HUAN LUYEN VA DANH GIA 15 TRUONG HOP
# ============================================================

all_results = []
classification_reports = []
feature_summaries = []

for scenario, description in SCENARIOS.items():

    print("\n" + "=" * 70)
    print("KICH BAN:", scenario)
    print(description)
    print("=" * 70)

    X_train = make_features(X_train_raw, scenario)
    X_test = make_features(X_test_raw, scenario)

    # Dam bao train va test co cung tap cot
    X_test = X_test.reindex(
        columns=X_train.columns,
        fill_value=0,
    )

    # Ep cac cot thanh so
    X_train = X_train.apply(
        pd.to_numeric,
        errors="coerce",
    )

    X_test = X_test.apply(
        pd.to_numeric,
        errors="coerce",
    )

    feature_summaries.append({
        "Ten_Thu_Nghiem": scenario,
        "Mo_Ta": description,
        "So_Luong_Feature": len(X_train.columns),
        "Danh_Sach_Feature": ", ".join(X_train.columns),
    })

    print("So feature:", len(X_train.columns))

    for model_name, pipeline_template in MODELS.items():

        print("\nDang huan luyen:", model_name)

        model = clone(pipeline_template)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # Lay score cho ROC-AUC
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.decision_function(X_test)

        # Cac metric danh gia
        accuracy = accuracy_score(y_test, y_pred)

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0,
        )

        roc_auc = roc_auc_score(
            y_test,
            y_score,
        )

        result = {
            "Ten_Thu_Nghiem": scenario,
            "Mo_Ta": description,
            "Mo_Hinh": model_name,
            "So_Luong_Feature": len(X_train.columns),
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "ROC_AUC": roc_auc,
        }

        all_results.append(result)

        print(
            f"Accuracy={accuracy:.4f} | "
            f"Precision={precision:.4f} | "
            f"Recall={recall:.4f} | "
            f"F1={f1:.4f} | "
            f"ROC-AUC={roc_auc:.4f}"
        )

        # Classification report
        report = classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0,
        )

        for label, values in report.items():
            if isinstance(values, dict):
                classification_reports.append({
                    "Ten_Thu_Nghiem": scenario,
                    "Mo_Hinh": model_name,
                    "Class": label,
                    **values,
                })

        # Ten file
        safe_model_name = (
            model_name.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
        )

        # Luu model cung danh sach cot dau vao
        artifact = {
            "model": model,
            "features": X_train.columns.tolist(),
            "scenario": scenario,
            "model_name": model_name,
            "target": TARGET_COL,
            "zero_as_missing": ZERO_AS_MISSING,
        }

        joblib.dump(
            artifact,
            MODEL_DIR / f"{scenario}_{safe_model_name}.joblib",
        )

        # Confusion matrix
        fig, ax = plt.subplots(figsize=(5, 4))

        ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix(
                y_test,
                y_pred,
                labels=[0, 1],
            ),
            display_labels=[0, 1],
        ).plot(
            ax=ax,
            cmap="Blues",
            colorbar=False,
        )

        ax.set_title(f"{model_name} - {scenario}")
        fig.tight_layout()

        fig.savefig(
            PLOT_DIR
            / f"confusion_matrix_{scenario}_{safe_model_name}.png",
            dpi=150,
            bbox_inches="tight",
        )

        plt.close(fig)

        # ROC curve
        fig, ax = plt.subplots(figsize=(6, 5))

        RocCurveDisplay.from_predictions(
            y_test,
            y_score,
            name=model_name,
            ax=ax,
        )

        ax.set_title(f"ROC Curve - {scenario}")
        fig.tight_layout()

        fig.savefig(
            PLOT_DIR
            / f"roc_curve_{scenario}_{safe_model_name}.png",
            dpi=150,
            bbox_inches="tight",
        )

        plt.close(fig)


# ============================================================
# 9. LUU BANG KET QUA
# ============================================================

results_df = pd.DataFrame(all_results)

results_df = results_df.sort_values(
    by=["ROC_AUC", "F1", "Recall"],
    ascending=False,
).reset_index(drop=True)

results_df.to_csv(
    OUTPUT_DIR / "baseline_model_results.csv",
    index=False,
    encoding="utf-8-sig",
)

feature_df = pd.DataFrame(feature_summaries)

feature_df.to_csv(
    OUTPUT_DIR / "feature_experiment_results.csv",
    index=False,
    encoding="utf-8-sig",
)

report_df = pd.DataFrame(classification_reports)

report_df.to_csv(
    OUTPUT_DIR / "classification_report.csv",
    index=False,
    encoding="utf-8-sig",
)

# Bang tong hop trung binh 3 mo hinh theo tung scenario
feature_summary = (
    results_df
    .groupby("Ten_Thu_Nghiem", as_index=False)
    .agg(
        So_Luong_Feature=("So_Luong_Feature", "first"),
        Accuracy_Trung_Binh=("Accuracy", "mean"),
        Recall_Trung_Binh=("Recall", "mean"),
        F1_Trung_Binh=("F1", "mean"),
        ROC_AUC_Trung_Binh=("ROC_AUC", "mean"),
    )
)

feature_summary.to_csv(
    OUTPUT_DIR / "feature_experiment_summary.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 10. VE BIEU DO SO SANH CAC MO HINH
# ============================================================

metrics = [
    ("Accuracy", "accuracy"),
    ("Recall", "recall"),
    ("F1", "f1"),
    ("ROC_AUC", "roc_auc"),
]

for metric_col, file_suffix in metrics:

    pivot = results_df.pivot(
        index="Ten_Thu_Nghiem",
        columns="Mo_Hinh",
        values=metric_col,
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(13, 6),
        rot=20,
    )

    ax.set_title(f"So sanh cac mo hinh - {metric_col}")
    ax.set_xlabel("Feature Engineering scenario")
    ax.set_ylabel(metric_col)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Model")

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / f"comparison_{file_suffix}.png",
        dpi=160,
        bbox_inches="tight",
    )

    plt.show()
    plt.close()


# ============================================================
# 11. LUU CAU HINH THUC NGHIEM
# ============================================================

config = {
    "dataset_path": str(DATA_PATH),
    "target_column": TARGET_COL,
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE,
    "train_rows": int(len(X_train_raw)),
    "test_rows": int(len(X_test_raw)),
    "original_features": BASE_FEATURES,
    "zero_as_missing": ZERO_AS_MISSING,
    "models": list(MODELS.keys()),
    "scenarios": SCENARIOS,
}

with open(
    OUTPUT_DIR / "experiment_config.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        config,
        f,
        ensure_ascii=False,
        indent=4,
    )


# ============================================================
# 12. HIEN THI KET QUA
# ============================================================

display_columns = [
    "Ten_Thu_Nghiem",
    "Mo_Hinh",
    "So_Luong_Feature",
    "Accuracy",
    "Recall",
    "F1",
    "ROC_AUC",
]

print("\n" + "=" * 100)
print("KET QUA DANH GIA 3 MO HINH")
print("=" * 100)

display(
    results_df[display_columns].round(4)
)

print("\n" + "=" * 100)
print("TOP 5 THEO ROC-AUC")
print("=" * 100)

display(
    results_df[display_columns].head(5).round(4)
)

best_result = results_df.iloc[0]

print("\nMO HINH CO ROC-AUC CAO NHAT TREN CAC THU NGHIEM")
print("Feature scenario:", best_result["Ten_Thu_Nghiem"])
print("Model:", best_result["Mo_Hinh"])
print(f"Accuracy: {best_result['Accuracy']:.4f}")
print(f"Recall:   {best_result['Recall']:.4f}")
print(f"F1-score: {best_result['F1']:.4f}")
print(f"ROC-AUC:  {best_result['ROC_AUC']:.4f}")


# ============================================================
# 13. DANH SACH FILE DA LUU
# ============================================================

print("\n" + "=" * 70)
print("HOAN THANH VIEC 13")
print("=" * 70)

print("\nThu muc ket qua:")
print(OUTPUT_DIR)

print("\nFile CSV:")
for path in sorted(OUTPUT_DIR.glob("*.csv")):
    print(" -", path.name)

print("\nFile bieu do:")
for path in sorted(PLOT_DIR.glob("*.png")):
    print(" -", path.name)

print("\nFile model:")
for path in sorted(MODEL_DIR.glob("*.joblib")):
    print(" -", path.name)
