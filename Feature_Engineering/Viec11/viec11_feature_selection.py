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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("viec11_audit.log", encoding='utf-8'), logging.StreamHandler()])
logger = logging.getLogger("Viec11_Selection")

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, cols): self.cols = cols
    def fit(self, X, y=None): return self
    def transform(self, X):
        df = pd.DataFrame(X, columns=self.cols)
        df['Glucose_to_Insulin_ratio'] = df['Glucose'] / df['Insulin'].clip(lower=1e-6)
        df['BMI_category'] = pd.cut(df['BMI'], bins=[0, 18.5, 25.0, 30.0, float('inf')], labels=['BMI_Low', 'BMI_Normal', 'BMI_High', 'BMI_VeryHigh'], right=False)
        df['Age_group'] = pd.cut(df['Age'], bins=[0, 35, 55, float('inf')], labels=['Age_Young', 'Age_Middle', 'Age_Senior'], right=False)
        df['BMI_category'] = pd.Categorical(df['BMI_category'], categories=['BMI_Low', 'BMI_Normal', 'BMI_High', 'BMI_VeryHigh'])
        df['Age_group'] = pd.Categorical(df['Age_group'], categories=['Age_Young', 'Age_Middle', 'Age_Senior'])
        return pd.get_dummies(df, columns=['BMI_category', 'Age_group'], drop_first=True)

def main():
    logger.info("BẮT ĐẦU CÔNG VIỆC 11: SO SÁNH PHƯƠNG PHÁP BẰNG F1, ROC-AUC, PR-AUC")
    path_to_viec7 = '../../Tien_Xu_Li/Viec_7/'
    X_train_raw = pd.read_csv(f'{path_to_viec7}X_train.csv')
    y_train = pd.read_csv(f'{path_to_viec7}y_train.csv').squeeze()
    base_cols = X_train_raw.columns.tolist()

    def evaluate_selection(selector_name, selector_obj=None):
        logger.info(f"Đánh giá: {selector_name}")
        steps = [('imputer', SimpleImputer(strategy='median')), ('fe', FeatureEngineer(cols=base_cols))]
        if selector_obj is not None: steps.append(('selector', selector_obj))
        steps.extend([('scaler', StandardScaler()), ('model', LogisticRegression(max_iter=1000, random_state=42))])
        
        pipeline = Pipeline(steps)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scoring = ['f1', 'roc_auc', 'average_precision']
        cv_results = cross_validate(pipeline, X_train_raw, y_train, cv=cv, scoring=scoring)
        
        pipeline.fit(X_train_raw, y_train)
        fe_cols = pipeline.named_steps['fe'].transform(pd.DataFrame(np.zeros((1, len(base_cols))), columns=base_cols)).columns
        kept_features = ", ".join(fe_cols[pipeline.named_steps['selector'].get_support()].tolist()) if selector_obj else "Tất cả"
            
        return {
            'Phương_Pháp': selector_name,
            'F1_Mean': round(cv_results['test_f1'].mean(), 4),
            'ROC_AUC_Mean': round(cv_results['test_roc_auc'].mean(), 4),
            'PR_AUC_Mean': round(cv_results['test_average_precision'].mean(), 4),
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
    
    best_idx = df_results['F1_Mean'].idxmax()
    best_pipeline = df_results.iloc[best_idx]['Pipeline']
    best_method = df_results.iloc[best_idx]['Phương_Pháp']
    
    logger.info(f"KẾT LUẬN: Phương pháp vô địch dựa trên F1-Score là [{best_method}]")
    
    df_results.drop(columns=['Pipeline']).to_csv('ket_luan_feature_selection.csv', index=False, encoding='utf-8-sig')
    feature_only_steps = best_pipeline.steps[:-1] 
    feature_only_pipeline = Pipeline(feature_only_steps)
    joblib.dump(feature_only_pipeline, 'data_preprocessing_pipeline_only.pkl')
    logger.info("Đã xuất file pkl.")

if __name__ == "__main__":
    main()