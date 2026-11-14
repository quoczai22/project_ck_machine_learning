# Kịch bản demo (2–3 phút)

1. **Giới thiệu dữ liệu:** Pima Indians Diabetes, 768 mẫu, 8 đặc trưng đầu vào, nhãn `Outcome` (1 là lớp dương). Số 0 ở Glucose, BloodPressure, SkinThickness, Insulin, BMI được coi là thiếu.
2. **Giới thiệu pipeline và model:** mở `reports/run_metadata.json`: feature order (8 gốc + `Glucose_BMI_Ratio`), tham số SVM đã chọn bằng CV trên Train (C=0,1, class_weight=balanced, gamma=scale), seed 42, ngưỡng 0,03 của quy tắc chọn.
3. **Chạy ứng dụng:** `streamlit run app/streamlit_app.py`; dùng giá trị mặc định đã điền sẵn.
4. **Dự đoán:** bấm **Dự đoán**, trình bày xác suất thuộc lớp Outcome = 1 và nhãn dự đoán. Giải thích vì sao nhãn lấy từ `predict()` còn xác suất qua Platt scaling nên hai đại lượng có thể không trùng ngưỡng 50%.
5. **Giá trị biên và số 0:** nhập 0 vào Glucose hoặc BMI, giải thích pipeline coi là thiếu và điền bằng trung vị của tập Train.
6. **Kiểm thử:** `python -m pytest tests/test_app.py`; nêu các trường hợp: giá trị mặc định, biên dưới và trên, khớp với pipeline đã lưu, số 0, thiếu model.
7. **Kết quả đánh giá:** mở `reports/error_analysis.md`: Recall 0,759 trên Test, FN và FP; nhắc Test đã được xem nhiều lần nên đây không phải đánh giá độc lập.
8. **Nêu giới hạn:** demo học tập, dataset nhỏ và hẹp, không dùng thay chẩn đoán y tế.

## Lời thoại mẫu

“Project dùng 8 đặc trưng để dự đoán nhãn Outcome. Dữ liệu được chia 80/20 có phân tầng với seed cố định. Pipeline gồm đổi số 0 thành thiếu, điền trung vị, thêm tỷ lệ Glucose/BMI, chuẩn hóa và SVM; tham số được chọn chỉ bằng cross-validation trên tập Train. Ứng dụng hiển thị xác suất và nhãn dự đoán. Kết quả chỉ phục vụ học tập, không thay thế chẩn đoán y tế.”
