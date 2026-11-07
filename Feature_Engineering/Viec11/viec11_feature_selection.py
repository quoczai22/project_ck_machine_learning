import pandas as pd
import numpy as np
import logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from functools import partial
import joblib
import warnings
warnings.filterwarnings('ignore')

# THIẾT LẬP AUDIT LOG
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("viec11_audit.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Viec11_Selection")

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        df = pd.DataFrame(X, columns=self.cols)
        df['Glucose_to_Insulin_ratio'] = df['Glucose'] / df['Insulin'].clip(lower=1e-6)
        bins_bmi = [0, 18.5, 25.0, 30.0, float('inf')]
        bmi_labels = ['Underweight', 'Normal', 'Overweight', 'Obese']
        df['BMI_category'] = pd.cut(df['BMI'], bins=bins_bmi, labels=bmi_labels, right=False)
        bins_age = [0, 35, 55, float('inf')]
        age_labels = ['Young_Adults', 'Middle_Aged', 'Seniors']
        df['Age_group'] = pd.cut(df['Age'], bins=bins_age, labels=age_labels, right=False)
        df['BMI_category'] = pd.Categorical(df['BMI_category'], categories=bmi_labels)
        df['Age_group'] = pd.Categorical(df['Age_group'], categories=age_labels)
        return pd.get_dummies(df, columns=['BMI_category', 'Age_group'], drop_first=True)

def main():
    logger.info("BẮT ĐẦU CÔNG VIỆC 11: SO SÁNH PHƯƠNG PHÁP CHỌN LỌC ĐẶC TRƯNG")
    
    path_to_viec7 = '../../Tien_Xu_Li/Viec_7/'
    X_train_raw = pd.read_csv(f'{path_to_viec7}X_train.csv')
    y_train = pd.read_csv(f'{path_to_viec7}y_train.csv').squeeze()
    base_cols = X_train_raw.columns.tolist()

    def evaluate_selection(selector_name, selector_obj=None):
        logger.info(f"Đánh giá phương pháp: {selector_name}")
        steps = [
            ('imputer', SimpleImputer(strategy='median')),
            ('fe', FeatureEngineer(cols=base_cols))
        ]
        if selector_obj is not None:
            steps.append(('selector', selector_obj))
            
        steps.extend([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(max_iter=1000, random_state=42))
        ])
        
        pipeline = Pipeline(steps)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_results = cross_validate(pipeline, X_train_raw, y_train, cv=cv, scoring=['recall', 'accuracy'])
        
        # Fit 1 lần để lấy tên cột
        pipeline.fit(X_train_raw, y_train)
        fe_columns = pipeline.named_steps['fe'].transform(pd.DataFrame(np.zeros((1, len(base_cols))), columns=base_cols)).columns
        
        if selector_obj is not None:
            mask = pipeline.named_steps['selector'].get_support()
            kept_features = ", ".join(fe_columns[mask].tolist())
        else:
            kept_features = "Tất cả"
            
        return {
            'Phương_Pháp': selector_name,
            'Recall_Mean': round(cv_results['test_recall'].mean(), 4),
            'Recall_Std': round(cv_results['test_recall'].std(), 4),
            'Feature_Được_Chọn': kept_features,
            'Pipeline': pipeline
        }

    results = []
    K_FEATURES = 8 
    mi_scorer = partial(mutual_info_classif, random_state=42)

    results.append(evaluate_selection("Không chọn (Giữ toàn bộ)"))
    results.append(evaluate_selection(f"f_classif (Chọn top {K_FEATURES})", SelectKBest(score_func=f_classif, k=K_FEATURES)))
    results.append(evaluate_selection(f"mutual_info (Chọn top {K_FEATURES})", SelectKBest(score_func=mi_scorer, k=K_FEATURES)))

    df_results = pd.DataFrame(results)
    
    # Tìm model có Recall cao nhất
    best_idx = df_results['Recall_Mean'].idxmax()
    best_pipeline = df_results.iloc[best_idx]['Pipeline']
    best_method = df_results.iloc[best_idx]['Phương_Pháp']
    
    logger.info(f"Kết luận: Phương pháp tốt nhất là [{best_method}]")
    
    df_results.drop(columns=['Pipeline']).to_csv('ket_luan_feature_selection.csv', index=False, encoding='utf-8-sig')
    
    # BÀN GIAO PIPELINE DẠNG PICKLE
    logger.info("Đang đóng gói Pipeline thành đối tượng .pkl để bàn giao...")
    joblib.dump(best_pipeline, 'feature_pipeline_for_app.pkl')
    logger.info("Đã xuất file: feature_pipeline_for_app.pkl")
    
    logger.info("HOÀN THÀNH CÔNG VIỆC 11.")

if __name__ == "__main__":
    main()