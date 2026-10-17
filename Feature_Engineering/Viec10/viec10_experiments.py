import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore') # Ẩn các cảnh báo không cần thiết

# 1. Đọc dữ liệu từ thư mục của Thành viên 2
path_to_viec7 = '../../Tien_Xu_Li/Viec_7/'
X_train_raw = pd.read_csv(f'{path_to_viec7}X_train.csv')
X_test_raw = pd.read_csv(f'{path_to_viec7}X_test.csv')
y_train = pd.read_csv(f'{path_to_viec7}y_train.csv').squeeze()
y_test = pd.read_csv(f'{path_to_viec7}y_test.csv').squeeze()

# 2. Xử lý giá trị NaN TRƯỚC KHI tạo Feature
# Học trung vị (median) từ tập Train, áp dụng cho cả Train và Test để tránh Data Leakage
imputer = SimpleImputer(strategy='median')
X_train = pd.DataFrame(imputer.fit_transform(X_train_raw), columns=X_train_raw.columns)
X_test = pd.DataFrame(imputer.transform(X_test_raw), columns=X_test_raw.columns)

# 3. Hàm tạo các Feature mới
def apply_feature_engineering(df):
    df_out = df.copy()
    
    # Feature 1: Glucose to Insulin Ratio
    # (Vì Insulin đã được điền NaN bằng median nên chắc chắn > 0, không lo lỗi chia 0)
    df_out['Glucose_to_Insulin_ratio'] = df_out['Glucose'] / df_out['Insulin']
    
    # Feature 2: BMI Category
    bins_bmi = [0, 18.5, 24.9, 29.9, float('inf')]
    labels_bmi = ['Underweight', 'Normal', 'Overweight', 'Obese']
    df_out['BMI_category'] = pd.cut(df_out['BMI'], bins=bins_bmi, labels=labels_bmi, right=False)
    
    # Feature 3: Age Group
    bins_age = [0, 35, 55, float('inf')]
    labels_age = ['Young_Adults', 'Middle_Aged', 'Seniors']
    df_out['Age_group'] = pd.cut(df_out['Age'], bins=bins_age, labels=labels_age, right=False)
    
    # Chuyển đổi các cột chữ (Category) thành cột số (One-Hot Encoding) để đưa vào mô hình
    df_out = pd.get_dummies(df_out, columns=['BMI_category', 'Age_group'], drop_first=True)
    
    return df_out

# Áp dụng hàm tạo feature cho cả Train và Test
X_train_fe = apply_feature_engineering(X_train)
X_test_fe = apply_feature_engineering(X_test)

# 4. Hàm để chạy 1 Thử nghiệm (Experiment)
def run_experiment(exp_name, train_data, test_data, feature_cols, removed_cols="Không có"):
    # Chỉ chọn các cột tham gia thử nghiệm
    X_tr = train_data[feature_cols]
    X_te = test_data[feature_cols]
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_te_scaled = scaler.transform(X_te)
    
    # Huấn luyện
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_tr_scaled, y_train)
    
    # Đánh giá
    acc = accuracy_score(y_test, model.predict(X_te_scaled))
    
    return {
        'Tên_Thử_Nghiệm': exp_name,
        'Accuracy': round(acc, 4),
        'Số_Lượng_Feature': len(feature_cols),
        'Feature_Bị_Loại': removed_cols
    }

# 5. TIẾN HÀNH CÁC THỬ NGHIỆM (EXPERIMENTS)
experiments_log = []

# Lấy danh sách cột gốc
base_cols = X_train_raw.columns.tolist()
# Lấy danh sách các cột sinh ra từ Category (đã mã hóa)
cat_cols = [col for col in X_train_fe.columns if 'BMI_category' in col or 'Age_group' in col]

# Thử nghiệm 1: Baseline (8 gốc - đã làm ở STT 9)
experiments_log.append(run_experiment(
    "1. Baseline", X_train, X_test, base_cols
))

# Thử nghiệm 2: Baseline + Tỷ lệ Ratio
exp2_cols = base_cols + ['Glucose_to_Insulin_ratio']
experiments_log.append(run_experiment(
    "2. Thêm Ratio", X_train_fe, X_test_fe, exp2_cols
))

# Thử nghiệm 3: Baseline + BMI & Age categories
exp3_cols = base_cols + cat_cols
experiments_log.append(run_experiment(
    "3. Thêm Category (BMI, Age)", X_train_fe, X_test_fe, exp3_cols
))

# Thử nghiệm 4: Dùng TẤT CẢ tính năng mới
all_cols = X_train_fe.columns.tolist()
experiments_log.append(run_experiment(
    "4. Tất cả Feature", X_train_fe, X_test_fe, all_cols
))

# Thử nghiệm 5: Tất cả Feature + LOẠI BỎ SkinThickness (ví dụ một feature thường bị nhiễu)
exp5_cols = [col for col in all_cols if col != 'SkinThickness']
experiments_log.append(run_experiment(
    "5. Tất cả Feature (Loại SkinThickness)", X_train_fe, X_test_fe, exp5_cols, removed_cols="SkinThickness"
))

# 6. Xuất báo cáo kết quả
results_df = pd.DataFrame(experiments_log)
print("\n=== BẢNG SO SÁNH FEATURE EXPERIMENTS ===")
print(results_df.to_string(index=False))

# Lưu file bàn giao
output_file = 'bang_so_sanh_feature_experiments.csv'
results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\nĐã lưu kết quả thành công vào file: {output_file}")