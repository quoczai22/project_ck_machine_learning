# Diabetes Prediction — Pima Indians

Project độc lập dự đoán nhãn `Outcome` của Pima Indians Diabetes Database. Không dùng mã nguồn, dữ liệu hay kết quả Telco Churn.

## Reproducible workflow

```powershell
cd diabetes-prediction
python -m src.training
pytest
streamlit run app/streamlit_app.py
```

`python -m src.training` chia Train/Test theo `stratify=Outcome, random_state=42`, so sánh ba nhóm mô hình bằng GridSearchCV với `StratifiedKFold`, đánh giá candidate cuối trên Test và lưu pipeline ở `models/diabetes_pipeline.joblib`. Lệnh này không tạo lại EDA.

## Chống leakage và xử lý dữ liệu

- `Outcome` được tách hẳn trước khi train.
- Các số 0 ở `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI` được đổi thành `NaN`.
- Median imputer, feature engineering, scaler và classifier đều ở trong cùng sklearn `Pipeline`; chúng chỉ `fit` trên các fold Train của GridSearch hoặc Train split.
- Bộ feature đã chốt: 8 feature gốc và 1 ratio `Glucose_BMI_Ratio` tạo sau impute, mẫu số BMI được clip tối thiểu `1e-6`. Bộ này theo cấu hình của Nguyễn Hiệu (scenario 2); so sánh với bộ 14 feature của Bảo và bộ 10 feature trước đó nằm trong độ lệch chuẩn CV.
- Model đã chốt: SVM (RBF), `C=0.1`, `class_weight='balanced'`, `gamma='scale'`. Lưới SVM trong `src/training.py` chỉ chứa cấu hình này, theo biên bản chốt (`docs/giao_viec/Bien_ban_chot_nguong_0_03.txt`).
- Outcome=1 là lớp dương. Chọn candidate bằng CV trên Train: F1 và ROC-AUC phải cách giá trị tốt nhất tương ứng không quá 0,03; trong các candidate đạt điều kiện ưu tiên Recall, sau đó F1 và ROC-AUC. Ngưỡng 0,03 đã được nhóm chốt, nhưng kết quả nhạy với ngưỡng: cấu hình trên chỉ được chọn khi ngưỡng ≥ 0,028. Test chỉ đánh giá candidate cuối, không dùng để chọn model. Test đã được tính trên nhiều cấu hình trong quá trình làm việc, nên không còn là đánh giá độc lập chưa từng quan sát.
- Protocol metric: Recall lớp dương là metric chính; Precision, F1, ROC-AUC, PR-AUC (average precision), Accuracy là bổ sung (ghi trong `metric_protocol` của metadata).
- Test được đánh giá một lần cho candidate đã chọn: `reports/model_comparison.csv` (metric + TN/FP/FN/TP), `reports/figures/confusion_matrix_<model>.png`, `reports/figures/roc_pr_curves_<model>.png`, `reports/error_profile_test.csv` (trung vị feature theo TN/FP/FN/TP) và `reports/error_analysis.md` (bảng metric, so sánh Train-CV với Test, phân tích FN/FP, giới hạn).
- Nhãn dự đoán lấy từ `predict()`. Với SVM, `predict()` dùng decision function còn xác suất là Platt scaling nên một số mẫu có xác suất < 0,5 vẫn được dự đoán 1; số mẫu này được ghi trong metadata và `error_analysis.md`.
- `reports/gridsearch_cv_results.csv` lưu mean và độ lệch chuẩn của các metric chọn model qua 5 fold. `reports/run_metadata.json` ghi lại quy tắc chọn, model và tham số được chọn kèm CV mean/std, kích thước split, nguồn và SHA-256 file dữ liệu; ngày tải cần được điền khi nhóm xác minh được.

## Artifacts tạo sau khi chạy

- `reports/figures/`: phân phối Outcome, histogram feature, correlation heatmap và confusion matrix. Các confusion matrix cũ của model khác có thể còn từ lần chạy trước; chỉ matrix của model được chọn trong metadata là kết quả hiện hành.
- `reports/zero_as_missing_summary.csv`, `reports/model_comparison.csv`, `reports/gridsearch_cv_results.csv`, `reports/run_metadata.json`.
- `models/diabetes_pipeline.joblib`: artifact được Streamlit nạp trực tiếp; không tái huấn luyện trong app. Nếu file không có, app hiển thị thông báo và yêu cầu chạy `python -m src.training`.
- `reports/run_metadata.json` có `feature_order` (thứ tự 9 cột đầu vào của pipeline) và `final_params`.
- `docs/logs/`: log kiểm tra leakage (chạy trên máy, xem `docs/logs/README_logs.md`).

## Ứng dụng

`streamlit run app/streamlit_app.py` mở giao diện nhập 8 feature gốc (không cần tự tính ratio). Giao diện có giới hạn giá trị cho từng ô (ví dụ Age 1–120, BMI 0–80), nạp pipeline SVM đã chốt từ `models/diabetes_pipeline.joblib`, hiển thị xác suất và lớp dự đoán, kèm cảnh báo không phải chẩn đoán y khoa. Giá trị 0 ở Glucose, BloodPressure, SkinThickness, Insulin, BMI được pipeline coi là thiếu. Nhãn lấy từ `predict()` còn xác suất qua Platt scaling, nên một số mẫu có xác suất dưới 50% vẫn được dự đoán lớp 1; app có ghi chú điều này.

Kiểm thử app: `python -m pytest tests/test_app.py` (7 test, mất khoảng 2 phút vì mỗi lần nạp lại model). Các trường hợp: render đủ 8 ô, giá trị mặc định, giá trị biên dưới và trên, khớp với pipeline đã lưu, toàn số 0 ở cột lâm sàng, và thiếu model (hiện thông báo lỗi, không văng ngoại lệ). Kịch bản demo ở `docs/demo_script.md`.

## Kiểm thử

Chạy `python -m pytest`. Các test kiểm tra zero-missing, ratio không chia 0, schema không nhận `Outcome`, imputer/scaler chỉ fit trên Train (Test bị làm hỏng cũng không đổi tham số đã fit), output dự đoán, nhất quán metric–confusion matrix (gồm ROC-AUC/PR-AUC), quy tắc chọn model chỉ nhận cột Train-CV, và tính nhất quán giữa model đã lưu, metadata, bảng CV và metric Test. Test nhất quán artifact sẽ fail nếu sửa code mà chưa chạy lại `python -m src.training`.
