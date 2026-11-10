# ============================================================
# VIEC 13 - BASELINE MODEL EVALUATION (TRAIN ONLY - 5-FOLD CV)
# Logistic Regression | Random Forest | SVM | DummyClassifier
# Dataset: Pima Indians Diabetes
# PHUONG PHAP: Stratified 5-Fold Cross-Validation tren tap TRAIN
# TUYET DOI KHONG DUNG TAP TEST DE DANH GIA HOAC CHON FEATURE
# ============================================================

import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

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
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. CAU HINH DU AN VA THU MUC LUU
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5

try:
    CURRENT_DIR = Path(__file__).resolve().parent
except NameError:
    CURRENT_DIR = Path.cwd().resolve()


def find_project_root(start):
    """
    Tim thu muc goc co Feature_Engineering hoac Model.
    """
    for p in [start, *start.parents]:
        if (
            (p / "Feature_Engineering").is_dir()
            or (p / "Model").is_dir()
        ):
            return p

    return start


PROJECT_ROOT = find_project_root(CURRENT_DIR)
OUTPUT_DIR = PROJECT_ROOT / "Model" / "Viec13"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = OUTPUT_DIR / "models"
PLOT_DIR = OUTPUT_DIR / "plots"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("VIEC 13 - BASELINE MODEL EVALUATION (TRAIN ONLY - 5-FOLD CV)")
print("=" * 70)
print("Project root:", PROJECT_ROOT)
print("Output directory:", OUTPUT_DIR)


# ============================================================
# 2. TIM VA DOC FILE DATASET
# ============================================================

def read_csv_robust(path):
    for kwargs in [
        {"sep": None, "engine": "python"},
        {"sep": ","},
        {"sep": ";"},
        {"sep": "\t"},
    ]:
        try:
            data = pd.read_csv(path, **kwargs)
            if data.shape[1] > 1:
                return data
        except Exception:
            continue
    raise ValueError(f"Khong doc duoc file CSV: {path}")


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
    preferred = [
        PROJECT_ROOT / "Tien_Xu_Li" / "Viec_5" / "pima_indians_zero_as_missing_NaN.csv",
        PROJECT_ROOT / "Tien_Xu_Li" / "pima-indians-diabetes.csv",
        PROJECT_ROOT / "Model" / "Tien_Xu_Li" / "pima-indians-diabetes.csv",
        PROJECT_ROOT / "Model" / "pima-indians-diabetes.csv",
    ]
    for path in preferred:
        if path.is_file():
            return path

    for root in [PROJECT_ROOT, CURRENT_DIR]:
        for path in root.rglob("*.csv"):
            if any(part.lower() in {".git", ".venv", "venv", "__pycache__"} for part in path.parts):
                continue
            cols = {normalize_column_name(c) for c in pd.read_csv(path, nrows=2).columns}
            if "glucose" in cols and "bmi" in cols and bool(cols.intersection({"class", "outcome", "target"})):
                return path

    raise FileNotFoundError("Khong tim thay dataset Pima Indians Diabetes.")


DATA_PATH = find_dataset()
df = read_csv_robust(DATA_PATH)
df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]

print("\nDataset duoc su dung:", DATA_PATH)
print("Kich thuoc ban dau:", df.shape)


# ============================================================
# 3. CHUAN HOA COT VA XAC DINH TARGET
# ============================================================

rename_map = {}
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

for col in df.columns:
    norm = normalize_column_name(col)
    if norm in aliases:
        rename_map[col] = aliases[norm]

df = df.rename(columns=rename_map)

TARGET_CANDIDATES = ["Class", "Outcome", "Diabetes", "Target"]
TARGET_COL = next((c for c in TARGET_CANDIDATES if c in df.columns), None)

if TARGET_COL is None:
    raise ValueError("Khong tim thay cot nhan.")

BASE_FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]

for col in BASE_FEATURES + [TARGET_COL]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=[TARGET_COL]).copy()
df = df[df[TARGET_COL].isin([0, 1])].copy()
df[TARGET_COL] = df[TARGET_COL].astype(int)
df = df[BASE_FEATURES + [TARGET_COL]].copy()

print("Cot target:", TARGET_COL)
print("Phan bo target toan bo dataset:")
print(df[TARGET_COL].value_counts().sort_index())


# ============================================================
# 4. XU LY CAC GIA TRI 0 BAT THUONG THEO QUY UOC PIMA
# ============================================================

ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in ZERO_AS_MISSING:
    df.loc[df[col] == 0, col] = np.nan

df.describe().to_csv(OUTPUT_DIR / "dataset_summary.csv", encoding="utf-8-sig")
df.isna().sum().rename("Missing_Count").to_csv(OUTPUT_DIR / "missing_values.csv", encoding="utf-8-sig")


# ============================================================
# 5. CHIA TRAIN VA TEST - NIEM PHONG TAP TEST
# ============================================================

X = df[BASE_FEATURES].copy()
y = df[TARGET_COL].copy()

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print(f"\nChia tap: Train = {X_train_raw.shape}, Test = {X_test_raw.shape}")
print(f"Phan bo lop tren TRAIN: 0 = {sum(y_train == 0)}, 1 = {sum(y_train == 1)} (Ty le duong = {y_train.mean():.2%})")
print(f"Phan bo lop tren TEST : 0 = {sum(y_test == 0)}, 1 = {sum(y_test == 1)} (Ty le duong = {y_test.mean():.2%})")
print(">> NGUYEN TAC: Tap Test duoc NIEM PHONG hoan toan. Viec 13 chi tinh toan tren tap TRAIN bang 5-Fold CV!")


# ============================================================
# 6. PROTOCOL METRIC CONFIRMATION (PCCV VIEC 17)
# ============================================================

protocol_info = {
    "Positive_Class": "Outcome = 1 (Nguoi mac benh tieu duong)",
    "Negative_Class": "Outcome = 0 (Nguoi khong mac benh tieu duong)",
    "Primary_Metric": "Recall (Do nhay lam sang - uu tien hang dau de tranh bo sot benh / False Negative)",
    "Secondary_Metrics": [
        "Accuracy (Do chinh xac tong the)",
        "Precision (Kiem soat ty le du doan duong tinh that)",
        "F1-Score (Trung binh dieu hoa giua Precision va Recall)",
        "ROC-AUC (Kha nang phan biet xac suat tong the)",
        "PR-AUC (Average Precision - danh gia tot tren du lieu mat can bang lop)",
    ],
    "Nguoi_Xac_Nhan_Modeling": "Nguyen Tan Hieu",
    "Nguoi_Xac_Nhan_Evaluation": "Kien Quoc",
    "Ngay_Xac_Nhan": "2026-11-08",
    "Trang_Thai": "DA CHOT PROTOCOL",
}

with open(OUTPUT_DIR / "protocol_metric_confirmation.json", "w", encoding="utf-8") as f:
    json.dump(protocol_info, f, ensure_ascii=False, indent=4)


# ============================================================
# 7. BASELINE MODELS CHUA TUNING - 5-FOLD CV TREN TRAIN (CONG 5)
# ============================================================

print("\n" + "=" * 70)
print("7. DANH GIA BASELINE MODEL CHUA TUNING (5-FOLD CV TREN TRAIN)")
print("=" * 70)

BASELINE_MODELS = {
    "Dummy (Most Frequent)": Pipeline([
        ("model", DummyClassifier(strategy="most_frequent")),
    ]),
    "Dummy (Stratified)": Pipeline([
        ("model", DummyClassifier(strategy="stratified", random_state=RANDOM_STATE)),
    ]),
    "Logistic Regression (Default)": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(random_state=RANDOM_STATE, max_iter=2000)),
    ]),
    "Random Forest (Default)": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(random_state=RANDOM_STATE)),
    ]),
    "SVM (Default)": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
    ]),
}

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
baseline_cv_summary = []
baseline_fold_records = []

for b_name, b_pipe in BASELINE_MODELS.items():
    f_acc, f_prec, f_rec, f_f1, f_auc, f_prauc = [], [], [], [], [], []

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X_train_raw, y_train), 1):
        X_tr, X_val = X_train_raw.iloc[tr_idx], X_train_raw.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        pipe = clone(b_pipe)
        pipe.fit(X_tr, y_tr)

        y_p = pipe.predict(X_val)
        if hasattr(pipe, "predict_proba"):
            y_s = pipe.predict_proba(X_val)[:, 1]
        elif hasattr(pipe, "decision_function"):
            y_s = pipe.decision_function(X_val)
        else:
            y_s = y_p

        acc = accuracy_score(y_val, y_p)
        prec = precision_score(y_val, y_p, zero_division=0)
        rec = recall_score(y_val, y_p, zero_division=0)
        f1 = f1_score(y_val, y_p, zero_division=0)

        try:
            auc = roc_auc_score(y_val, y_s)
        except Exception:
            auc = 0.5

        try:
            prauc = average_precision_score(y_val, y_s)
        except Exception:
            prauc = y_val.mean()

        f_acc.append(acc)
        f_prec.append(prec)
        f_rec.append(rec)
        f_f1.append(f1)
        f_auc.append(auc)
        f_prauc.append(prauc)

        baseline_fold_records.append({
            "Mo_Hinh": b_name,
            "Fold": fold,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "ROC_AUC": auc,
            "PR_AUC": prauc,
        })

    baseline_cv_summary.append({
        "Mo_Hinh": b_name,
        "Loai": "Baseline chua tuning (8 features goc)",
        "Du_Lieu": "TRAIN (5-Fold CV)",
        "Accuracy": float(np.mean(f_acc)),
        "Accuracy_Std": float(np.std(f_acc)),
        "Precision": float(np.mean(f_prec)),
        "Precision_Std": float(np.std(f_prec)),
        "Recall": float(np.mean(f_rec)),
        "Recall_Std": float(np.std(f_rec)),
        "F1": float(np.mean(f_f1)),
        "F1_Std": float(np.std(f_f1)),
        "ROC_AUC": float(np.mean(f_auc)),
        "ROC_AUC_Std": float(np.std(f_auc)),
        "PR_AUC": float(np.mean(f_prauc)),
        "PR_AUC_Std": float(np.std(f_prauc)),
    })

baseline_df = pd.DataFrame(baseline_cv_summary)
baseline_df.to_csv(OUTPUT_DIR / "baseline_model_results.csv", index=False, encoding="utf-8-sig")
baseline_df.to_csv(OUTPUT_DIR / "baseline_cv_train_summary.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(baseline_fold_records).to_csv(OUTPUT_DIR / "baseline_cv_train_per_fold.csv", index=False, encoding="utf-8-sig")

print("\nBang Baseline Models (5-Fold CV tren Train):")
cols_view = ["Mo_Hinh", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
print(baseline_df[cols_view].round(4).to_string(index=False))


# ============================================================
# 8. FEATURE ENGINEERING - 5 KICH BAN (VIEC 9)
# ============================================================

SCENARIOS = {
    "1_Baseline": "8 feature goc",
    "2_Them_Ratio": "Them Glucose_BMI_Ratio",
    "3_Them_Category": "Them Ratio va Category BMI/Age",
    "4_Tat_Ca_Feature": "Them Ratio va Category va feature phu",
    "5_Bo_SkinThickness": "Feature engineered, bo SkinThickness",
}


def make_features(X_input, scenario):
    Z = X_input[BASE_FEATURES].copy()

    if scenario in ["2_Them_Ratio", "3_Them_Category", "4_Tat_Ca_Feature", "5_Bo_SkinThickness"]:
        Z["Glucose_BMI_Ratio"] = Z["Glucose"] / Z["BMI"].replace(0, np.nan)

    if scenario in ["3_Them_Category", "4_Tat_Ca_Feature", "5_Bo_SkinThickness"]:
        Z["BMI_Category"] = pd.cut(
            Z["BMI"],
            bins=[-np.inf, 18.5, 25, 30, np.inf],
            labels=["Underweight", "Normal", "Overweight", "Obese"],
        ).astype("object")

        Z["Age_Category"] = pd.cut(
            Z["Age"],
            bins=[-np.inf, 30, 45, np.inf],
            labels=["Young", "Middle", "Older"],
        ).astype("object")

    if scenario in ["4_Tat_Ca_Feature", "5_Bo_SkinThickness"]:
        Z["Glucose_Insulin_Ratio"] = Z["Glucose"] / Z["Insulin"].replace(0, np.nan)
        Z["Age_BMI_Ratio"] = Z["Age"] / Z["BMI"].replace(0, np.nan)

    if scenario == "5_Bo_SkinThickness":
        Z = Z.drop(columns=["SkinThickness"], errors="ignore")

    categorical_cols = [c for c in ["BMI_Category", "Age_Category"] if c in Z.columns]
    if categorical_cols:
        Z = pd.get_dummies(Z, columns=categorical_cols, dtype=int)

    Z = Z.replace([np.inf, -np.inf], np.nan)
    return Z


# ============================================================
# 9. HUAN LUYEN VA DANH GIA 15 TRUONG HOP BANG 5-FOLD CV TREN TRAIN
# (5 Scenarios x 3 Models = 15 truong hop)
# ============================================================

print("\n" + "=" * 70)
print("8. DANH GIA 15 KICH BAN THUC NGHIEM (5-FOLD CV TREN TRAIN)")
print("=" * 70)

MODELS = {
    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ]),
    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=1)),
    ]),
    "SVM": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
    ]),
}

experiment_results = []
experiment_fold_records = []
feature_summaries = []

for scenario, description in SCENARIOS.items():
    X_train_scen = make_features(X_train_raw, scenario)

    feature_summaries.append({
        "Ten_Thu_Nghiem": scenario,
        "Mo_Ta": description,
        "So_Luong_Feature": len(X_train_scen.columns),
        "Danh_Sach_Feature": ", ".join(X_train_scen.columns),
    })

    print(f"\n--- Kich ban: {scenario} ({len(X_train_scen.columns)} features) ---")

    for model_name, pipeline_template in MODELS.items():
        f_acc, f_prec, f_rec, f_f1, f_auc, f_prauc = [], [], [], [], [], []

        for fold, (tr_idx, val_idx) in enumerate(cv.split(X_train_scen, y_train), 1):
            X_tr, X_val = X_train_scen.iloc[tr_idx], X_train_scen.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

            model = clone(pipeline_template)
            model.fit(X_tr, y_tr)

            y_p = model.predict(X_val)
            y_s = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else (model.decision_function(X_val) if hasattr(model, "decision_function") else y_p)

            acc = accuracy_score(y_val, y_p)
            prec = precision_score(y_val, y_p, zero_division=0)
            rec = recall_score(y_val, y_p, zero_division=0)
            f1 = f1_score(y_val, y_p, zero_division=0)

            try:
                auc = roc_auc_score(y_val, y_s)
            except Exception:
                auc = 0.5

            try:
                prauc = average_precision_score(y_val, y_s)
            except Exception:
                prauc = y_val.mean()

            f_acc.append(acc)
            f_prec.append(prec)
            f_rec.append(rec)
            f_f1.append(f1)
            f_auc.append(auc)
            f_prauc.append(prauc)

            experiment_fold_records.append({
                "Ten_Thu_Nghiem": scenario,
                "Mo_Hinh": model_name,
                "Fold": fold,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1": f1,
                "ROC_AUC": auc,
                "PR_AUC": prauc,
            })

        mean_acc = float(np.mean(f_acc))
        mean_prec = float(np.mean(f_prec))
        mean_rec = float(np.mean(f_rec))
        mean_f1 = float(np.mean(f_f1))
        mean_auc = float(np.mean(f_auc))
        mean_prauc = float(np.mean(f_prauc))

        experiment_results.append({
            "Ten_Thu_Nghiem": scenario,
            "Mo_Ta": description,
            "Mo_Hinh": model_name,
            "So_Luong_Feature": len(X_train_scen.columns),
            "Du_Lieu": "TRAIN (5-Fold CV)",
            "Accuracy": mean_acc,
            "Accuracy_Std": float(np.std(f_acc)),
            "Precision": mean_prec,
            "Precision_Std": float(np.std(f_prec)),
            "Recall": mean_rec,
            "Recall_Std": float(np.std(f_rec)),
            "F1": mean_f1,
            "F1_Std": float(np.std(f_f1)),
            "ROC_AUC": mean_auc,
            "ROC_AUC_Std": float(np.std(f_auc)),
            "PR_AUC": mean_prauc,
            "PR_AUC_Std": float(np.std(f_prauc)),
        })

        print(
            f"  {model_name:20s} | "
            f"Acc: {mean_acc:.4f} | "
            f"Precision: {mean_prec:.4f} | "
            f"Recall: {mean_rec:.4f} | "
            f"F1: {mean_f1:.4f} | "
            f"ROC-AUC: {mean_auc:.4f}"
        )

        # Fit toan bo Train de luu artifact
        model_full_train = clone(pipeline_template)
        model_full_train.fit(X_train_scen, y_train)

        safe_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        joblib.dump(
            {
                "model": model_full_train,
                "features": X_train_scen.columns.tolist(),
                "scenario": scenario,
                "model_name": model_name,
                "evaluation": "5-Fold CV on Train",
            },
            MODEL_DIR / f"{scenario}_{safe_name}.joblib",
        )


# ============================================================
# 10. LUU FILE VA VE BIEU DO SO SANH 5 CHI SO TREN TRAIN
# ============================================================

exp_df = pd.DataFrame(experiment_results)
exp_df.to_csv(OUTPUT_DIR / "feature_experiment_results.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(feature_summaries).to_csv(OUTPUT_DIR / "feature_experiment_summary.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(experiment_fold_records).to_csv(OUTPUT_DIR / "feature_experiment_per_fold.csv", index=False, encoding="utf-8-sig")

# Tao classification report tong hop tu CV tren Train
class_report_rows = []
for row in experiment_results:
    class_report_rows.append({
        "Ten_Thu_Nghiem": row["Ten_Thu_Nghiem"],
        "Mo_Hinh": row["Mo_Hinh"],
        "Du_Lieu": "TRAIN (5-Fold CV)",
        "Precision": row["Precision"],
        "Recall": row["Recall"],
        "F1_Score": row["F1"],
        "Support_Positive": sum(y_train == 1),
        "Accuracy": row["Accuracy"],
        "ROC_AUC": row["ROC_AUC"],
    })
pd.DataFrame(class_report_rows).to_csv(OUTPUT_DIR / "classification_report.csv", index=False, encoding="utf-8-sig")

# Ve bieu do so sanh 5 metrics tren tap Train
metrics_to_plot = [
    ("Accuracy", "So sanh Accuracy (5-Fold CV tren Train)", "comparison_accuracy.png"),
    ("Precision", "So sanh Precision (5-Fold CV tren Train)", "comparison_precision.png"),
    ("Recall", "So sanh Recall (5-Fold CV tren Train)", "comparison_recall.png"),
    ("F1", "So sanh F1-Score (5-Fold CV tren Train)", "comparison_f1.png"),
    ("ROC_AUC", "So sanh ROC-AUC (5-Fold CV tren Train)", "comparison_roc_auc.png"),
]

for metric_col, title, filename in metrics_to_plot:
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot = exp_df.pivot(index="Ten_Thu_Nghiem", columns="Mo_Hinh", values=metric_col)
    pivot.plot(kind="bar", ax=ax, rot=15, colormap="viridis")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel(metric_col)
    ax.set_xlabel("Kich ban Feature")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(title="Mo hinh")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)

print("\n" + "=" * 70)
print("HOAN THANH VIEC 13: TAT CA METRIC DUOC TINH BANG 5-FOLD CV TREN TRAIN!")
print("Cac file da luu trong Model/Viec13:")
for p in sorted(OUTPUT_DIR.iterdir()):
    if p.is_file():
        print(" -", p.name)
print("=" * 70)
