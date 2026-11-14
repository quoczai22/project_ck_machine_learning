# Nội dung slide – Việc 21 đến 24

## Slide 1 – Phạm vi phụ trách
- Việc 21: đóng gói pipeline và metadata.
- Việc 22: xây dựng giao diện Streamlit.
- Việc 23: kiểm thử chức năng và giao diện.
- Việc 24: README, slide và kịch bản demo.

## Slide 2 – Candidate model đã chốt
- Scenario: 2_Them_Ratio.
- Model: SVM (RBF).
- Tham số: C=0.1, gamma=scale, class_weight=balanced, probability=True.
- Chọn bằng StratifiedKFold 5-fold + GridSearchCV trên Train; ưu tiên Recall và có ràng buộc F1/ROC-AUC.

## Slide 3 – Artifact và pipeline
- `pipeline.joblib`: model đã fit; không train lại khi chạy app.
- `metadata.json`: feature order, tham số, protocol, metric và quy tắc tiền xử lý.
- Pipeline: SimpleImputer(median) → StandardScaler → SVC(RBF).

## Slide 4 – Luồng xử lý đầu vào
1. Nhập 8 feature gốc.
2. 0 ở 5 cột sinh lý → NaN; Pregnancies giữ nguyên.
3. Tạo Glucose_BMI_Ratio = Glucose / BMI.
4. Đưa 9 feature theo đúng thứ tự vào pipeline.
5. Lấy nhãn bằng predict(), xác suất bằng predict_proba().

## Slide 5 – Giao diện Streamlit
- Form 8 trường nhập.
- Nút Dự đoán.
- Hiển thị Outcome và xác suất lớp 1.
- Cảnh báo giá trị thiếu và disclaimer học tập.

## Slide 6 – Kiểm thử
- Test load artifact.
- Test feature order và ratio.
- Test zero-as-missing và giữ Pregnancies=0.
- Test dữ liệu thiếu/ngoài khoảng.
- Test end-to-end; checklist trình duyệt cần được chạy và ký xác nhận thực tế.

## Slide 7 – Kết quả và giới hạn
| Metric | CV Train | Test đã ghi nhận |
|---|---:|---:|
| Accuracy | 0.7296 | 0.6688 |
| Precision | 0.5835 | 0.5190 |
| Recall | 0.7947 | 0.7593 |
| F1 | 0.6713 | 0.6165 |
| ROC-AUC | 0.8303 | 0.7831 |
- Review nhóm ghi nhận Test đã được xem nhiều lần.
- Chỉ phục vụ minh họa học máy; không phải công cụ chẩn đoán.
