import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# 1. ĐỌC VÀ CHUẨN BỊ DỮ LIỆU (Giống STT 10)
path_to_viec7 = '../../Tien_Xu_Li/Viec_7/'
X_train_raw = pd.read_csv(f'{path_to_viec7}X_train.csv')
X_test_raw = pd.read_csv(f'{path_to_viec7}X_test.csv')
y_train = pd.read_csv(f'{path_to_viec7}y_train.csv').squeeze()
y_test = pd.read_csv(f'{path_to_viec7}y_test.csv').squeeze()

# Điền NaN bằng Median
imputer = SimpleImputer(strategy='median')
X_train = pd.DataFrame(imputer.fit_transform(X_train_raw), columns=X_train_raw.columns)
X_test = pd.DataFrame(imputer.transform(X_test_raw), columns=X_test_raw.columns)

# Tạo Feature mới
def apply_feature_engineering(df):
    df_out = df.copy()
    df_out['Glucose_to_Insulin_ratio'] = df_out['Glucose'] / df_out['Insulin']
    bins_bmi = [0, 18.5, 24.9, 29.9, float('inf')]
    df_out['BMI_category'] = pd.cut(df_out['BMI'], bins=bins_bmi, labels=['Underweight', 'Normal', 'Overweight', 'Obese'], right=False)
    bins_age = [0, 35, 55, float('inf')]
    df_out['Age_group'] = pd.cut(df_out['Age'], bins=bins_age, labels=['Young_Adults', 'Middle_Aged', 'Seniors'], right=False)
    df_out = pd.get_dummies(df_out, columns=['BMI_category', 'Age_group'], drop_first=True)
    return df_out

X_train_fe = apply_feature_engineering(X_train)
X_test_fe = apply_feature_engineering(X_test)

# 2. HÀM ĐÁNH GIÁ PIPELINE FEATURE SELECTION
# Sử dụng Pipeline để đảm bảo Scaler -> SelectKBest -> Model không bị rò rỉ dữ liệu
def evaluate_selection(selector_name, selector_obj=None):
    steps = [('scaler', StandardScaler())]
    
    if selector_obj is not None:
        steps.append(('selector', selector_obj))
        
    steps.append(('model', LogisticRegression(max_iter=1000, random_state=42)))
    
    pipeline = Pipeline(steps)
    pipeline.fit(X_train_fe, y_train)
    y_pred = pipeline.predict(X_test_fe)
    acc = accuracy_score(y_test, y_pred)
    
    # Lấy danh sách các feature được giữ lại (nếu có dùng selector)
    kept_features = "Tất cả"
    if selector_obj is not None:
        mask = pipeline.named_steps['selector'].get_support()
        kept_features = ", ".join(X_train_fe.columns[mask].tolist())
        
    return {
        'Phương_Pháp': selector_name,
        'Accuracy': round(acc, 4),
        'Feature_Được_Chọn': kept_features
    }

# 3. THỰC HIỆN CÁC THỬ NGHIỆM SO SÁNH
results = []
# K = 8 là số lượng feature mong muốn giữ lại (bạn có thể thay đổi để test)
K_FEATURES = 8 

# Kịch bản 1: Không chọn lọc (Dùng toàn bộ data sau Feature Engineering)
results.append(evaluate_selection("Không chọn (Giữ toàn bộ)"))

# Kịch bản 2: Dùng f_classif
results.append(evaluate_selection(
    f"f_classif (Chọn top {K_FEATURES})", 
    SelectKBest(score_func=f_classif, k=K_FEATURES)
))

# Kịch bản 3: Dùng mutual_info_classif
results.append(evaluate_selection(
    f"mutual_info (Chọn top {K_FEATURES})", 
    SelectKBest(score_func=mutual_info_classif, k=K_FEATURES)
))

# 4. IN VÀ XUẤT KẾT QUẢ BÀN GIAO
df_results = pd.DataFrame(results)

# Tìm phương pháp tốt nhất
best_method = df_results.loc[df_results['Accuracy'].idxmax()]

print("\n=== BẢNG SO SÁNH FEATURE SELECTION ===")
print(df_results.to_string(index=False))

print(f"\n=> KẾT LUẬN: Phương pháp cho điểm cao nhất là '{best_method['Phương_Pháp']}' với Accuracy = {best_method['Accuracy']}.")
print(f"=> Nhóm nên sử dụng tập feature sau: {best_method['Feature_Được_Chọn']}")

# Lưu file báo cáo
output_csv = 'ket_luan_feature_selection.csv'
df_results.to_csv(output_csv, index=False, encoding='utf-8-sig')
print(f"\nĐã xuất kết luận ra file: {output_csv}")