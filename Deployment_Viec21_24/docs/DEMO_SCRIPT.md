# Kịch bản demo – Việc 24 (khoảng 3–4 phút)

**Mở đầu**

“Em phụ trách phần đóng gói mô hình, xây dựng ứng dụng Streamlit, kiểm thử và tài liệu demo. Ứng dụng sử dụng mô hình đã được nhóm lựa chọn ở Việc 15, không huấn luyện lại khi khởi động.”

**Giới thiệu artifact**

“Artifact `pipeline.joblib` là pipeline đã fit gồm SimpleImputer với median, StandardScaler và SVM kernel RBF. Metadata đi kèm lưu model, tham số, thứ tự feature, quy tắc tiền xử lý và metric. Mô hình được chốt là scenario 2, bổ sung Glucose_BMI_Ratio cho 8 biến gốc.”

**Demo ứng dụng**

“Ở giao diện, người dùng nhập 8 đặc trưng. Theo quy tắc của nhóm, số 0 ở Glucose, BloodPressure, SkinThickness, Insulin và BMI được hiểu là dữ liệu thiếu; riêng Pregnancies bằng 0 vẫn hợp lệ. Ứng dụng tạo thêm Glucose_BMI_Ratio, sắp xếp feature đúng như artifact rồi đưa vào pipeline.”

“Sau khi nhấn Dự đoán, ứng dụng hiển thị nhãn Outcome 0 hoặc 1 và xác suất ước lượng cho lớp 1. Nhãn lấy trực tiếp từ `predict()` của SVM; xác suất chỉ để tham khảo, không dùng ngưỡng 0,5 tự đặt để ghi đè nhãn.”

**Kiểm thử**

“Em chạy pytest để kiểm tra nạp artifact, thứ tự 9 feature, công thức ratio, xử lý giá trị 0, kiểm tra dữ liệu và dự đoán end-to-end. Ngoài ra, checklist thủ công có các bước kiểm tra form, cảnh báo, tải model và thao tác trên trình duyệt.”

**Kết thúc**

“Các chỉ số Test được giữ nguyên theo kết quả Việc 15: Accuracy 0,6688; Precision 0,5190; Recall 0,7593; F1 0,6165; ROC-AUC 0,7831. Biên bản review cũng ghi nhận tập Test đã được xem nhiều lần, nên nhóm không xem đây là lần đánh giá Test đầu tiên. Ứng dụng chỉ phục vụ minh họa học máy, không thay thế chẩn đoán y khoa.”

## Câu hỏi phản biện có thể gặp

**1. Vì sao không train lại khi mở app?**

Vì mục tiêu của bước triển khai là sử dụng đúng candidate đã được nhóm chọn và fit. Train lại có thể tạo ra mô hình khác và làm sai lệch kết quả đã chốt.

**2. Vì sao phải lưu cả pipeline thay vì chỉ lưu SVM?**

SVM cần dữ liệu đã được impute và scale. Lưu cả pipeline giúp áp dụng đúng các bước đã fit trên Train cho dữ liệu mới, tránh xử lý khác nhau giữa lúc train và inference.

**3. Vì sao có 9 feature trong khi form chỉ có 8?**

Form nhận 8 biến gốc. Ứng dụng tự tạo feature thứ 9 `Glucose_BMI_Ratio` theo scenario 2 đã chọn.

**4. Vì sao Outcome=1 không đơn giản là xác suất >= 0,5?**

Mã đánh giá của nhóm dùng `SVC.predict()` để lấy nhãn. Với SVC bật probability, `predict_proba` là xác suất hiệu chỉnh riêng; không nên thay quy tắc nhãn đã dùng trong đánh giá bằng một threshold tùy ý.

**5. Kết quả có thể dùng chẩn đoán không?**

Không. Đây là mô hình học tập trên dataset công khai, có giới hạn về dữ liệu và khả năng khái quát; kết quả không thay thế đánh giá y khoa.
