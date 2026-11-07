import pandas as pd
import numpy as np
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
import os

# =====================================================================
# THIẾT LẬP AUDIT LOG (NHẬT KÝ KIỂM TOÁN)
# =====================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("viec9_audit.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Viec9_Baseline")

def main():
    logger.info("BẮT ĐẦU CÔNG VIỆC 9: THIẾT LẬP BASELINE (8 FEATURE GỐC)")
    
    # 1. NẠP DỮ LIỆU (CHỈ DÙNG TRAIN SET)
    path_to_viec7 = '../../Tien_Xu_Li/Viec_7/'
    try:
        logger.info(f"Đang đọc dữ liệu Train từ: {path_to_viec7}")
        X_train = pd.read_csv(f'{path_to_viec7}X_train.csv')
        y_train = pd.read_csv(f'{path_to_viec7}y_train.csv').squeeze()
        logger.info(f"Đã nạp thành công X_train với shape: {X_train.shape}")
    except FileNotFoundError:
        logger.error("Không tìm thấy file X_train.csv. Vui lòng kiểm tra lại đường dẫn.")
        return

    # Ghi log không vi phạm Data Leakage
    logger.info("Hành động: Đã cách ly hoàn toàn X_test và y_test theo yêu cầu chống rò rỉ dữ liệu.")

    # 2. XÂY DỰNG PIPELINE
    logger.info("Khởi tạo Pipeline: SimpleImputer (median) -> StandardScaler -> LogisticRegression")
    baseline_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=1000, random_state=42))
    ])

    # 3. ĐÁNH GIÁ BẰNG CROSS-VALIDATION
    logger.info("Bắt đầu chạy Stratified 5-Fold Cross-Validation trên tập Train...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Tính cả Recall (chính) và Accuracy (phụ)
    cv_results = cross_validate(baseline_pipeline, X_train, y_train, cv=cv, scoring=['recall', 'accuracy'])
    
    mean_recall = cv_results['test_recall'].mean()
    std_recall = cv_results['test_recall'].std()
    mean_acc = cv_results['test_accuracy'].mean()
    std_acc = cv_results['test_accuracy'].std()
    
    logger.info(f"Hoàn thành CV. Metric chính [Recall]: {mean_recall:.4f} ± {std_recall:.4f}")
    logger.info(f"Hoàn thành CV. Metric phụ [Accuracy]: {mean_acc:.4f} ± {std_acc:.4f}")

    # 4. XUẤT BẢNG METRIC
    logger.info("Đang xuất bảng tổng hợp Metric CV...")
    metric_df = pd.DataFrame({
        'Metric': ['Recall', 'Accuracy'], 
        'Mean_CV': [mean_recall, mean_acc], 
        'Std_CV': [std_recall, std_acc]
    })
    metric_df.to_csv('bang_metric_cv_baseline.csv', index=False)
    logger.info("Đã xuất file: bang_metric_cv_baseline.csv")

    # 5. XUẤT BẢNG TRỌNG SỐ (FEATURE IMPORTANCE)
    logger.info("Đang huấn luyện (fit) lại Pipeline trên toàn bộ X_train để lấy hệ số (weights)...")
    baseline_pipeline.fit(X_train, y_train)
    
    baseline_features = pd.DataFrame({
        'Feature': X_train.columns,
        'Baseline_Weight': baseline_pipeline.named_steps['model'].coef_[0]
    })
    baseline_features['Absolute_Weight'] = baseline_features['Baseline_Weight'].abs()
    baseline_features = baseline_features.sort_values(by='Absolute_Weight', ascending=False).drop(columns=['Absolute_Weight'])

    baseline_features.to_csv('bang_baseline_feature_set.csv', index=False)
    logger.info("Đã xuất file: bang_baseline_feature_set.csv")
    
    logger.info("HOÀN THÀNH CÔNG VIỆC 9.")

if __name__ == "__main__":
    main()