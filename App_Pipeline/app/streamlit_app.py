"""Streamlit UI using the persisted sklearn Pipeline (SVM chosen by Train CV)."""
from pathlib import Path
import json
import sys
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.inference import load_model, predict
from src.preprocessing import FEATURE_COLUMNS, ZERO_AS_MISSING

MODEL_PATH = ROOT / "models" / "diabetes_pipeline.joblib"
METADATA_PATH = ROOT / "reports" / "run_metadata.json"

# (label, min, max, default, step) — integer fields use ints so Streamlit keeps them integral.
FIELDS = {
    "Pregnancies": ("Pregnancies — Số lần mang thai", 0, 20, 1, 1),
    "Glucose": ("Glucose", 0.0, 300.0, 120.0, 1.0),
    "BloodPressure": ("BloodPressure", 0.0, 200.0, 70.0, 1.0),
    "SkinThickness": ("SkinThickness", 0.0, 100.0, 20.0, 1.0),
    "Insulin": ("Insulin", 0.0, 900.0, 80.0, 1.0),
    "BMI": ("BMI", 0.0, 80.0, 28.0, 0.1),
    "DiabetesPedigreeFunction": ("DiabetesPedigreeFunction", 0.0, 3.0, 0.5, 0.01),
    "Age": ("Age — Tuổi", 1, 120, 33, 1),
}

st.set_page_config(page_title="Dự đoán nguy cơ tiểu đường", page_icon="🩺", layout="centered")
st.title("Dự đoán tiểu đường — Pima Indians Diabetes")
st.caption("Demo học tập. Kết quả không phải chẩn đoán y khoa.")

if not MODEL_PATH.exists():
    st.error(f"Không tìm thấy model đã lưu: {MODEL_PATH}. Chạy `python -m src.training` để tạo lại.")
    st.stop()

model = load_model(MODEL_PATH)
metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8")) if METADATA_PATH.exists() else {}
if metadata:
    st.caption(f"Model: {metadata.get('final_model', 'SVM')} — tham số: {metadata.get('final_params', {})}")

st.info(
    "Nhập 8 đặc trưng bên dưới. Với "
    + ", ".join(ZERO_AS_MISSING)
    + ", giá trị 0 được pipeline coi là thiếu và thay bằng trung vị của tập huấn luyện."
)

with st.form("prediction_form"):
    left, right = st.columns(2)
    inputs = {}
    for index, column in enumerate(FEATURE_COLUMNS):
        label, low, high, default, step = FIELDS[column]
        with (left if index < 4 else right):
            inputs[column] = st.number_input(label, min_value=low, max_value=high, value=default, step=step)
    submitted = st.form_submit_button("Dự đoán", type="primary")

if submitted:
    output = predict(model, pd.DataFrame([inputs])).iloc[0]
    st.subheader("Kết quả")
    st.metric("Xác suất thuộc lớp Outcome = 1", f"{output.diabetes_probability:.1%}")
    if int(output.prediction) == 1:
        st.warning("Mô hình dự đoán: Outcome = 1")
    else:
        st.success("Mô hình dự đoán: Outcome = 0")
    st.caption(
        "Nhãn lấy từ `predict()` của SVM (decision function), còn xác suất qua Platt scaling; "
        "hai đại lượng có thể không trùng ngưỡng 50% (một số mẫu có xác suất dưới 50% vẫn được dự đoán lớp 1)."
    )
    st.caption("Đây là kết quả mô hình thống kê, không thay thế tư vấn hoặc chẩn đoán của nhân viên y tế.")
