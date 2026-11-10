import pandas as pd
import numpy as np
import logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("viec10_audit.log", encoding='utf-8'), logging.StreamHandler()])
logger = logging.getLogger("Viec10_Experiments")

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols
    def fit(self, X, y=None): return self
    def transform(self, X):
        df = pd.DataFrame(X, columns=self.cols)
        df['Glucose_to_Insulin_ratio'] = df['Glucose'] / df['Insulin'].clip(lower=1e-6)
        bins_bmi = [0, 18.5, 25.0, 30.0, float('inf')]
        bmi_labels = ['BMI_Low', 'BMI_Normal', 'BMI_High', 'BMI_VeryHigh']
        df['BMI_category'] = pd.cut(df['BMI'], bins=bins_bmi, labels=bmi_labels, right=False)
        bins_age = [0, 35, 55, float('inf')]
        age_labels = ['Age_Young', 'Age_Middle', 'Age_Senior']
        df['Age_group'] = pd.cut(df['Age'], bins=bins_age, labels=age_labels, right=False)
        df['BMI_category'] = pd.Categorical(df['BMI_category'], categories=bmi_labels)
        df['Age_group'] = pd.Categorical(df['Age_group'], categories=age_labels)
        return pd.get_dummies(df, columns=['BMI_category', 'Age_group'], drop_first=True)

class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_keep): self.cols_to_keep = cols_to_keep
    def fit(self, X, y=None): return self
    def transform(self, X):
        if self.cols_to_keep == 'all': return X
        return X[[c for c in X.columns if c in self.cols_to_keep]]

def main():
    logger.info("BẮT ĐẦU CÔNG VIỆC 10: THỬ NGHIỆM FEATURE VỚI METRIC F1, ROC-AUC, PR-AUC, RECALL")
    path_to_viec7 = '../../Tien_Xu_Li/Viec_7/'
    X_train_raw = pd.read_csv(f'{path_to_viec7}X_train.csv')
    y_train = pd.read_csv(f'{path_to_viec7}y_train.csv').squeeze()
    base_cols = X_train_raw.columns.tolist()
    
    def run_experiment(exp_name, features_to_keep, removed_cols="Không có"):
        logger.info(f"Đang chạy: {exp_name}")
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('fe', FeatureEngineer(cols=base_cols)),
            ('selector', ColumnSelector(features_to_keep)),
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(max_iter=1000, random_state=42))
        ])
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        # Bổ sung Recall
        scoring = ['f1', 'roc_auc', 'average_precision', 'recall']
        cv_results = cross_validate(pipeline, X_train_raw, y_train, cv=cv, scoring=scoring)
        
        return {
            'Tên_Thử_Nghiệm': exp_name,
            'F1_Mean': round(cv_results['test_f1'].mean(), 4),
            'F1_Std': round(cv_results['test_f1'].std(), 4),
            'ROC_AUC_Mean': round(cv_results['test_roc_auc'].mean(), 4),
            'ROC_AUC_Std': round(cv_results['test_roc_auc'].std(), 4),
            'PR_AUC_Mean': round(cv_results['test_average_precision'].mean(), 4),
            'PR_AUC_Std': round(cv_results['test_average_precision'].std(), 4),
            'Recall_Mean': round(cv_results['test_recall'].mean(), 4),
            'Recall_Std': round(cv_results['test_recall'].std(), 4),
            'Feature_Bị_Loại': removed_cols
        }

    dummy_fe = FeatureEngineer(cols=base_cols)
    dummy_df = dummy_fe.transform(pd.DataFrame(np.zeros((1, len(base_cols))), columns=base_cols))
    cat_cols = [c for c in dummy_df.columns if 'BMI_category' in c or 'Age_group' in c]
    all_cols = dummy_df.columns.tolist()

    experiments_log = []
    experiments_log.append(run_experiment("1. Baseline", base_cols))
    experiments_log.append(run_experiment("2. Thêm Ratio", base_cols + ['Glucose_to_Insulin_ratio']))
    experiments_log.append(run_experiment("3. Thêm Category (BMI, Age)", base_cols + cat_cols))
    experiments_log.append(run_experiment("4. Tất cả Feature", 'all'))
    experiments_log.append(run_experiment("5. Tất cả (Loại SkinThickness)", [c for c in all_cols if c != 'SkinThickness'], "SkinThickness"))

    pd.DataFrame(experiments_log).to_csv('bang_so_sanh_feature_experiments.csv', index=False, encoding='utf-8-sig')
    logger.info("Hoàn thành xuất file.")

if __name__ == "__main__":
    main()