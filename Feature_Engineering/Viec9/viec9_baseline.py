import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import os

# 1. Đường dẫn tới thư mục của Thành viên 2
path_to_viec7 = '../../Tien_Xu_Li/Viec_7/'

try:
    X_train = pd.read_csv(f'{path_to_viec7}X_train.csv')
    X_test = pd.read_csv(f'{path_to_viec7}X_test.csv')
    y_train = pd.read_csv(f'{path_to_viec7}y_train.csv').squeeze() 
    y_test = pd.read_csv(f'{path_to_viec7}y_test.csv').squeeze()
    print("8 feature gốc!")
except FileNotFoundError:
    print("Không tìm thấy file. Hãy kiểm tra lại đường dẫn.")
    exit()

# 2. Xây dựng Pipeline tiền xử lý và huấn luyện
# Pipeline sẽ tự động chạy theo thứ tự: Lấp NaN -> Chuẩn hóa -> Đưa vào LogisticRegression
baseline_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')), # Điền NaN bằng trung vị
    ('scaler', StandardScaler()),                  # Chuẩn hóa dữ liệu
    ('model', LogisticRegression(max_iter=1000, random_state=42))
])

# Huấn luyện toàn bộ Pipeline trên tập Train
baseline_pipeline.fit(X_train, y_train)

# 3. Đánh giá mô hình trên tập Test
y_pred = baseline_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n--- KẾT QUẢ ĐÁNH GIÁ BASELINE ---")
print(f"Accuracy: {accuracy:.4f}")
print("\nChi tiết Classification Report:")
print(classification_report(y_test, y_pred))

# 4. Xuất Bảng Baseline Feature Set
# Trích xuất mô hình Logistic Regression từ trong Pipeline ra để lấy trọng số
trained_model = baseline_pipeline.named_steps['model']

baseline_features = pd.DataFrame({
    'Feature': X_train.columns,
    'Baseline_Weight': trained_model.coef_[0]
})

# Sắp xếp theo mức độ ảnh hưởng tuyệt đối (để xem feature nào quan trọng nhất)
baseline_features['Absolute_Weight'] = baseline_features['Baseline_Weight'].abs()
baseline_features = baseline_features.sort_values(by='Absolute_Weight', ascending=False).drop(columns=['Absolute_Weight'])

print("\nBảng Baseline Feature Set (Sắp xếp theo độ quan trọng):")
print(baseline_features)

# Lưu kết quả bàn giao
output_csv = 'bang_baseline_feature_set.csv'
baseline_features.to_csv(output_csv, index=False)
print(f"\nĐã xuất file kết quả: {output_csv} tại thư mục hiện tại")