from pathlib import Path
import sys
import streamlit as st

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from prediction import load_artifact, predict_patient

st.set_page_config(page_title="Pima Diabetes Prediction", page_icon="🩺", layout="centered")
st.title("Dự đoán nguy cơ tiểu đường – Pima Indians Diabetes")
st.caption("Ứng dụng minh họa học máy cho mục đích học tập")
st.warning("Kết quả chỉ là dự đoán của mô hình trên dữ liệu Pima; không phải chẩn đoán y khoa và không thay thế tư vấn của nhân viên y tế.")

@st.cache_resource(show_spinner="Đang tải pipeline đã huấn luyện...")
def get_artifact():
    return load_artifact(HERE / "artifacts" / "pipeline.joblib")

try:
    artifact = get_artifact()
except Exception as exc:
    st.error(f"Không thể tải model artifact: {exc}")
    st.info("Kiểm tra file artifacts/pipeline.joblib và phiên bản thư viện trong requirements.txt.")
    st.stop()

st.markdown("Nhập đủ **8 đặc trưng** của bệnh nhân. Giá trị 0 ở Glucose, BloodPressure, SkinThickness, Insulin và BMI được xử lý như dữ liệu thiếu, theo quy tắc tiền xử lý của đồ án.")
with st.form("patient_form"):
    c1, c2 = st.columns(2)
    with c1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=30, value=2, step=1, help="Số lần mang thai; 0 là giá trị hợp lệ.")
        glucose = st.number_input("Glucose", min_value=0.0, max_value=400.0, value=120.0, step=1.0)
        blood_pressure = st.number_input("BloodPressure", min_value=0.0, max_value=250.0, value=70.0, step=1.0)
        skin_thickness = st.number_input("SkinThickness", min_value=0.0, max_value=150.0, value=20.0, step=1.0)
    with c2:
        insulin = st.number_input("Insulin", min_value=0.0, max_value=1500.0, value=80.0, step=1.0)
        bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=30.0, step=0.1)
        dpf = st.number_input("DiabetesPedigreeFunction", min_value=0.0, max_value=5.0, value=0.5, step=0.01, format="%.3f")
        age = st.number_input("Age", min_value=18, max_value=120, value=33, step=1)
    submitted = st.form_submit_button("Dự đoán", type="primary", use_container_width=True)

if submitted:
    patient = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
    }
    try:
        result = predict_patient(patient, artifact)
        st.subheader("Kết quả dự đoán")
        if result["outcome"] == 1:
            st.error("Mô hình dự đoán: Outcome = 1 (nhóm dương tính)")
        else:
            st.success("Mô hình dự đoán: Outcome = 0 (nhóm âm tính)")
        if result["probability"] is not None:
            st.metric("Xác suất lớp Outcome = 1 do SVM ước lượng", f"{result['probability']:.1%}")
            st.progress(min(max(result["probability"], 0.0), 1.0))
        if result["missing_fields_imputed"]:
            st.info("Các giá trị 0 sau được xem là thiếu và được impute bằng median đã học trong pipeline: " + ", ".join(result["missing_fields_imputed"]))
        st.caption("Nhãn dự đoán lấy trực tiếp từ model.predict(); không tự đặt ngưỡng 0,5 lên predict_proba để thay đổi nhãn.")
    except Exception as exc:
        st.error(f"Không thể dự đoán: {exc}")

with st.expander("Thông tin mô hình"):
    st.write(f"**Mô hình:** {artifact.get('model_name', 'SVM (RBF)')}")
    st.write(f"**Feature scenario:** {artifact.get('scenario', '2_Them_Ratio')}")
    st.write("**Pipeline:** SimpleImputer(median) → StandardScaler → SVC(RBF, probability=True)")
    st.write("**Feature đầu vào mô hình:** 8 feature gốc + Glucose_BMI_Ratio")
