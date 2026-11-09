import pandas as pd
import numpy as np
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate

# THIẾT LẬP AUDIT LOG
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
    
    path_to_viec7 = '../../Tien_Xu_Li/Viec_7/'
    try:
        X_train = pd.read_csv(f'{path_to_viec7}X_train.csv')
        y_train = pd.read_csv(f'{path_to_viec7}y_train.csv').squeeze()
    except FileNotFoundError:
        logger.error("Không tìm thấy file X_train.csv. Vui lòng kiểm tra lại đường dẫn.")
        return

    logger.info("Đã cách ly X_test/y_test. Khởi tạo Pipeline Baseline...")
    
    baseline_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=1000, random_state=42))
    ])

    # CẬP NHẬT METRIC MỚI THEO YÊU CẦU NT
    logger.info("Chạy Stratified 5-Fold CV với metric: F1, ROC-AUC, PR-AUC...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring_metrics = ['f1', 'roc_auc', 'average_precision'] # average_precision chính là PR-AUC
    
    cv_results = cross_validate(baseline_pipeline, X_train, y_train, cv=cv, scoring=scoring_metrics)
    
    mean_f1 = cv_results['test_f1'].mean()
    mean_roc = cv_results['test_roc_auc'].mean()
    mean_pr = cv_results['test_average_precision'].mean()
    
    logger.info(f"Kết quả CV - F1: {mean_f1:.4f} | ROC-AUC: {mean_roc:.4f} | PR-AUC: {mean_pr:.4f}")

    # XUẤT BẢNG METRIC
    metric_df = pd.DataFrame({
        'Metric': ['F1-Score', 'ROC-AUC', 'PR-AUC'], 
        'Mean_CV': [mean_f1, mean_roc, mean_pr], 
        'Std_CV': [cv_results['test_f1'].std(), cv_results['test_roc_auc'].std(), cv_results['test_average_precision'].std()]
    })
    metric_df.to_csv('bang_metric_cv_baseline.csv', index=False)
    
    # XUẤT TRỌNG SỐ
    baseline_pipeline.fit(X_train, y_train)
    baseline_features = pd.DataFrame({
        'Feature': X_train.columns,
        'Baseline_Weight': baseline_pipeline.named_steps['model'].coef_[0]
    })
    baseline_features['Absolute_Weight'] = baseline_features['Baseline_Weight'].abs()
    baseline_features = baseline_features.sort_values(by='Absolute_Weight', ascending=False).drop(columns=['Absolute_Weight'])
    baseline_features.to_csv('bang_baseline_feature_set.csv', index=False)
    
    logger.info("Đã lưu các file CSV.")

if __name__ == "__main__":
    main()