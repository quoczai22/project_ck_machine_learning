# Hướng dẫn cho Tùng — baseline ứng dụng DA 02

Thư mục này là bản baseline mà nhóm trưởng dựng sẵn cho phần artifact, ứng dụng và kiểm thử (PCCV việc 21–24). Tùng vẫn là người phụ trách; hai bạn làm chung trên nhánh `Tung`.

## Chạy thử
```bash
cd App_Pipeline
pip install -r requirements.txt
python -m pytest            # 20 test, khoảng 1–2 phút
streamlit run app/streamlit_app.py
```

## Baseline gồm gì
- `models/diabetes_pipeline.joblib`: SVM (C=0,1, balanced, gamma='scale'), 9 feature (8 gốc + Glucose/BMI), đã bao gồm bước đổi 0 thành NaN, median, scale. Chỉ tải artifact từ nguồn tin cậy.
- `reports/run_metadata.json`: `feature_order`, `final_params`, protocol metric, kích thước split, SHA-256 dữ liệu.
- `app/streamlit_app.py`: giao diện dựa trên bản của Tùng (form hai cột, giới hạn giá trị), nạp pipeline trên, có thông báo thiếu model.
- `tests/`: 20 test (pipeline, leakage, đánh giá, app).
- `docs/demo_script.md`: kịch bản demo theo model SVM.

## Điểm khác so với bản đầu của Tùng (đã sửa)
- Model đổi từ Logistic Regression (Recall Test 0,5) sang SVM đã chốt của nhóm (Recall Test 0,759).
- Bước 0 thành NaN nằm trong pipeline, app không còn tự xử lý.
- Thêm test cho giá trị biên, số 0 và thiếu model.

## Việc còn lại của Tùng
1. Chạy app trên trình duyệt, chụp ảnh các màn hình (nhập mặc định, kết quả, giá trị biên) cho báo cáo mục 9.
2. Lập checklist kiểm thử app chính thức (PCCV việc 23) và chạy các trường hợp bằng tay.
3. Làm slide thuyết trình và chạy thử kịch bản demo.
4. Viết phần ứng dụng trong báo cáo (nền tảng, chức năng, hướng dẫn, hình ảnh).
5. Xác nhận các thay đổi ở `app/` và `README.md` là phù hợp.

## Lưu ý khi viết báo cáo
- Tập Test đã được xem nhiều lần trong quá trình làm; không mô tả là đánh giá độc lập chưa từng quan sát.
- Nhãn dự đoán lấy từ `predict()` của SVM, còn xác suất qua Platt scaling; một số mẫu có xác suất dưới 50% vẫn được dự đoán lớp 1 (25/154 mẫu Test).
- Kết quả chỉ phục vụ học tập, không phải chẩn đoán y khoa.
