# Checklist kiểm thử ứng dụng – Việc 23

## A. Kiểm thử tự động

Chạy `pytest -q`. Các test bao gồm:

- [ ] Nạp được `pipeline.joblib` và đúng scenario / model đã chốt.
- [ ] Tạo đúng 9 cột và đúng thứ tự feature trong artifact.
- [ ] Tính `Glucose_BMI_Ratio` đúng.
- [ ] Chuyển 0 thành NaN ở đúng 5 cột sinh lý; giữ `Pregnancies=0`.
- [ ] Từ chối giá trị ngoài khoảng kiểm tra hoặc thiếu feature.
- [ ] Chạy dự đoán end-to-end và trả Outcome 0/1 cùng xác suất trong [0,1].

## B. Kiểm thử thủ công trên trình duyệt

| ID | Thao tác | Kết quả mong đợi | Đạt |
|---|---|---|---|
| UI-01 | Mở ứng dụng | Tiêu đề, cảnh báo học tập, form 8 trường hiển thị | ☐ |
| UI-02 | Nhấn Dự đoán với giá trị mặc định | Hiện Outcome 0 hoặc 1 và xác suất | ☐ |
| UI-03 | Nhập Pregnancies = 0 | Chấp nhận giá trị 0 | ☐ |
| UI-04 | Nhập Glucose = 0 | Hiện thông báo giá trị được xem là thiếu/impute | ☐ |
| UI-05 | Nhập BMI = 0 | Không phát sinh chia cho 0; pipeline xử lý NaN | ☐ |
| UI-06 | Nhập giá trị vượt giới hạn UI | Widget không cho vượt giới hạn | ☐ |
| UI-07 | Tải lại trang và dự đoán lần nữa | Model tải lại ổn định, không huấn luyện | ☐ |
| UI-08 | Tạm đổi tên artifact rồi tải app | Hiện thông báo thiếu model, không crash im lặng | ☐ |
| UI-09 | Mở phần Thông tin mô hình | Hiện đúng SVM RBF và scenario 2 | ☐ |
| UI-10 | Kiểm tra cảnh báo y khoa | Hiện disclaimer, không khẳng định chẩn đoán | ☐ |

**Biên bản chạy trình duyệt:** Người kiểm thử: __________  Ngày: __________  Hệ điều hành/trình duyệt: __________

Ghi chú lỗi và ảnh chụp màn hình: ____________________________________________

> Không đánh dấu các mục UI là đạt nếu chưa thực sự chạy trên máy/trình duyệt. Các test tự động chỉ xác nhận logic Python, không thay thế kiểm thử giao diện.
