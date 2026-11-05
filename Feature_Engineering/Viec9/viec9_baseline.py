import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
import os

# 1. Chỉ nạp tập Train
path_to_viec7 = '../../Tien_Xu_Li/Viec_7/'
try:
    X_train = pd.read_csv(f'{path_to_viec7}X_train.csv')
    y_train = pd.read_csv(f'{path_to_viec7}y_train.csv').squeeze() 
except FileNotFoundError:
    print("Không tìm thấy file. Hãy kiểm tra lại đường dẫn.")
    exit()

# 2. Xây dựng Pipeline
baseline_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000, random_state=42))
])

# 3. Đánh giá bằng CV trên Train (Metric: Recall)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
recall_scores = cross_val_score(baseline_pipeline, X_train, y_train, cv=cv, scoring='recall')

mean_recall = recall_scores.mean()
std_recall = recall_scores.std()

print(f"\n--- KẾT QUẢ ĐÁNH GIÁ BASELINE ---")
print(f"Recall: {mean_recall:.4f} ± {std_recall:.4f}")

# Xuất bảng Metric
metric_df = pd.DataFrame({'Metric': ['Recall'], 'Mean': [mean_recall], 'Std': [std_recall]})
metric_df.to_csv('bang_metric_cv_baseline.csv', index=False)

# 4. Xuất Bảng Trọng Số
baseline_pipeline.fit(X_train, y_train)
baseline_features = pd.DataFrame({
    'Feature': X_train.columns,
    'Baseline_Weight': baseline_pipeline.named_steps['model'].coef_[0]
})
baseline_features['Absolute_Weight'] = baseline_features['Baseline_Weight'].abs()
baseline_features = baseline_features.sort_values(by='Absolute_Weight', ascending=False).drop(columns=['Absolute_Weight'])

baseline_features.to_csv('bang_baseline_feature_set.csv', index=False)
print("Đã lưu bang_metric_cv_baseline.csv và bang_baseline_feature_set.csv")