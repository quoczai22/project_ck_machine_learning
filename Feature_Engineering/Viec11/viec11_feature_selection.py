import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from functools import partial
import warnings
warnings.filterwarnings('ignore')

# 1. Đọc và chuẩn bị dữ liệu Train (Giống file viec 10)
path_to_viec7 = '../../Tien_Xu_Li/Viec_7/'
X_train_raw = pd.read_csv(f'{path_to_viec7}X_train.csv')
y_train = pd.read_csv(f'{path_to_viec7}y_train.csv').squeeze()

imputer = SimpleImputer(strategy='median')
X_train = pd.DataFrame(imputer.fit_transform(X_train_raw), columns=X_train_raw.columns)

def apply_feature_engineering(df):
    df_out = df.copy()
    safe_insulin = df_out['Insulin'].clip(lower=1e-6)
    df_out['Glucose_to_Insulin_ratio'] = df_out['Glucose'] / safe_insulin
    df_out['BMI_category'] = pd.cut(df_out['BMI'], bins=[0, 18.5, 24.9, 29.9, float('inf')], labels=['Underweight', 'Normal', 'Overweight', 'Obese'], right=False)
    df_out['Age_group'] = pd.cut(df_out['Age'], bins=[0, 35, 55, float('inf')], labels=['Young_Adults', 'Middle_Aged', 'Seniors'], right=False)
    df_out = pd.get_dummies(df_out, columns=['BMI_category', 'Age_group'], drop_first=True)
    return df_out

X_train_fe = apply_feature_engineering(X_train)

# 2. Đánh giá Selector bằng Cross-Validation trên Train
def evaluate_selection(selector_name, selector_obj=None):
    steps = [('scaler', StandardScaler())]
    if selector_obj is not None:
        steps.append(('selector', selector_obj))
    steps.append(('model', LogisticRegression(max_iter=1000, random_state=42)))
    
    pipeline = Pipeline(steps)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X_train_fe, y_train, cv=cv, scoring='accuracy')
    
    kept_features = "Tất cả"
    if selector_obj is not None:
        # Fit 1 lần trên toàn bộ Train để lấy tên feature được chọn
        pipeline.fit(X_train_fe, y_train)
        mask = pipeline.named_steps['selector'].get_support()
        kept_features = ", ".join(X_train_fe.columns[mask].tolist())
        
    return {
        'Phương_Pháp': selector_name,
        'CV_Accuracy_Train': round(scores.mean(), 4),
        'Feature_Được_Chọn': kept_features
    }

results = []
K_FEATURES = 8 

# SỬA LỖI 2: Thêm random_state cho mutual_info_classif bằng partial
mi_scorer = partial(mutual_info_classif, random_state=42)

results.append(evaluate_selection("Không chọn (Giữ toàn bộ)"))
results.append(evaluate_selection(f"f_classif (Chọn top {K_FEATURES})", SelectKBest(score_func=f_classif, k=K_FEATURES)))
results.append(evaluate_selection(f"mutual_info (Chọn top {K_FEATURES})", SelectKBest(score_func=mi_scorer, k=K_FEATURES)))

df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))
df_results.to_csv('ket_luan_feature_selection.csv', index=False, encoding='utf-8-sig')