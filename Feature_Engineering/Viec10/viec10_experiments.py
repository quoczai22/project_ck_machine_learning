import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# 1. Đọc dữ liệu
path_to_viec7 = '../../Tien_Xu_Li/Viec_7/'
X_train_raw = pd.read_csv(f'{path_to_viec7}X_train.csv')
y_train = pd.read_csv(f'{path_to_viec7}y_train.csv').squeeze()

# 2. Xử lý giá trị NaN
imputer = SimpleImputer(strategy='median')
X_train = pd.DataFrame(imputer.fit_transform(X_train_raw), columns=X_train_raw.columns)

# 3. Hàm tạo Feature MỚI (Đã sửa lỗi chia 0)
def apply_feature_engineering(df):
    df_out = df.copy()
    
    # SỬA LỖI 3: Ép Insulin >= 1e-6 để chống chia 0 khi inference
    safe_insulin = df_out['Insulin'].clip(lower=1e-6)
    df_out['Glucose_to_Insulin_ratio'] = df_out['Glucose'] / safe_insulin
    
    bins_bmi = [0, 18.5, 24.9, 29.9, float('inf')]
    df_out['BMI_category'] = pd.cut(df_out['BMI'], bins=bins_bmi, labels=['Underweight', 'Normal', 'Overweight', 'Obese'], right=False)
    
    bins_age = [0, 35, 55, float('inf')]
    df_out['Age_group'] = pd.cut(df_out['Age'], bins=bins_age, labels=['Young_Adults', 'Middle_Aged', 'Seniors'], right=False)
    
    df_out = pd.get_dummies(df_out, columns=['BMI_category', 'Age_group'], drop_first=True)
    return df_out

X_train_fe = apply_feature_engineering(X_train)

# 4. Hàm chạy thử nghiệm (SỬA LỖI 1: Dùng CV trên Train, không chạm tới Test)
def run_experiment(exp_name, train_data, feature_cols, removed_cols="Không có"):
    X_tr = train_data[feature_cols]
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    # StratifiedKFold với 5 folds
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X_tr, y_train, cv=cv, scoring='accuracy')
    
    return {
        'Tên_Thử_Nghiệm': exp_name,
        'CV_Accuracy_Train': round(scores.mean(), 4), # Lấy điểm trung bình của 5 folds
        'Số_Lượng_Feature': len(feature_cols),
        'Feature_Bị_Loại': removed_cols
    }

# 5. TIẾN HÀNH CÁC THỬ NGHIỆM
experiments_log = []
base_cols = X_train_raw.columns.tolist()
cat_cols = [col for col in X_train_fe.columns if 'BMI_category' in col or 'Age_group' in col]

experiments_log.append(run_experiment("1. Baseline", X_train, base_cols))
experiments_log.append(run_experiment("2. Thêm Ratio", X_train_fe, base_cols + ['Glucose_to_Insulin_ratio']))
experiments_log.append(run_experiment("3. Thêm Category (BMI, Age)", X_train_fe, base_cols + cat_cols))
all_cols = X_train_fe.columns.tolist()
experiments_log.append(run_experiment("4. Tất cả Feature", X_train_fe, all_cols))
exp5_cols = [col for col in all_cols if col != 'SkinThickness']
experiments_log.append(run_experiment("5. Tất cả Feature (Loại SkinThickness)", X_train_fe, exp5_cols, "SkinThickness"))

results_df = pd.DataFrame(experiments_log)
print(results_df.to_string(index=False))
results_df.to_csv('bang_so_sanh_feature_experiments.csv', index=False, encoding='utf-8-sig')