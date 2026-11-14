# Đánh giá cuối trên Test và phân tích lỗi

Model: **SVM**, tham số `{'classifier__C': 0.1, 'classifier__class_weight': 'balanced', 'classifier__gamma': 'scale'}`, được chọn chỉ bằng 5-fold CV trên Train (xem `run_metadata.json`).
Outcome=1 là lớp dương. Dùng ngưỡng mặc định của estimator; không tinh chỉnh ngưỡng trên Test.

Nhãn dự đoán lấy từ `predict()` của estimator. Có 25/154 mẫu Test mà nhãn khác với quy tắc `predict_proba >= 0.5`; xác suất nhỏ nhất vẫn được dự đoán là 1 là 0.316. Với SVM, `predict()` dùng decision function còn xác suất là Platt scaling hiệu chỉnh riêng, nên hai đại lượng không trùng ngưỡng 0,5. Confusion matrix, Recall, Precision, F1, Accuracy dựa trên `predict()`; ROC-AUC và PR-AUC dựa trên xác suất.

## Metric trên Test (n=154: 54 dương, 100 âm)

| Metric | Giá trị |
|---|---|
| Recall (metric chính) | 0.7593 |
| Precision | 0.5190 |
| F1 | 0.6165 |
| ROC-AUC | 0.7824 |
| PR-AUC (average precision) | 0.6729 (mức ngẫu nhiên ≈ tỷ lệ dương 0.351) |
| Accuracy | 0.6688 |

Confusion matrix: TN=62, FP=38, FN=13, TP=41.
Hình: `figures/confusion_matrix_svm.png`, `figures/roc_pr_curves_svm.png`.

## Train-CV so với Test (chỉ để báo cáo, không dùng để chọn lại)

| Metric | Train-CV mean ± std | Test |
|---|---|---|
| recall | 0.7947 ± 0.0607 | 0.7593 |
| f1 | 0.6699 ± 0.0181 | 0.6165 |
| roc_auc | 0.8298 ± 0.0107 | 0.7824 |

## False Negative / False Positive

- FN=13: bỏ sót 13/54 mẫu dương (24.1%). Đây là lỗi protocol Recall ưu tiên giảm.
- FP=38: báo dương 38/100 mẫu âm (38.0%); hệ quả là kiểm tra thêm không cần thiết.
- Đánh đổi: trong 79 lần báo dương có 38 lần sai (Precision=0.519).

Trung vị feature theo nhóm (0 ở Glucose, BloodPressure, SkinThickness, Insulin, BMI được bỏ qua như giá trị thiếu):

| group | n | Pregnancies | Glucose | BloodPressure | SkinThickness | Insulin | BMI | DiabetesPedigreeFunction | Age | diabetes_probability |
|---|---|---|---|---|---|---|---|---|---|---|
| TP | 41 | 4 | 146 | 78 | 32 | 184 | 34.9 | 0.479 | 38 | 0.658 |
| FN | 13 | 4 | 114 | 65 | 30.5 | 101.5 | 32 | 0.258 | 29 | 0.136 |
| TN | 62 | 2 | 100 | 68 | 23 | 93 | 27.1 | 0.304 | 23 | 0.072 |
| FP | 38 | 4.5 | 133 | 77 | 34.5 | 194 | 34.85 | 0.316 | 35.5 | 0.51 |

- FN so với TP: Glucose: FN=114 so với TP=146; BMI: FN=32 so với TP=34.9; Age: FN=29 so với TP=38; diabetes_probability: FN=0.136 so với TP=0.658.
- FP so với TN: Glucose: FP=133 so với TN=100; BMI: FP=34.85 so với TN=27.1; Age: FP=35.5 so với TN=23; diabetes_probability: FP=0.51 so với TN=0.072.

Các so sánh trên chỉ mô tả 154 mẫu Test, nhóm FN/FP rất nhỏ; không phải quan hệ nhân quả.

## Giới hạn

- Test chỉ có 154 mẫu (54 dương): mỗi FN làm Recall thay đổi khoảng 0.019, nên ước lượng có sai số lớn.
- Dữ liệu là một quần thể hẹp (phụ nữ gốc Pima, từ 21 tuổi), nhiều giá trị thiếu ở Insulin/SkinThickness; không chứng minh khả năng tổng quát hóa.
- Kết quả phục vụ học tập, **không phải chẩn đoán y khoa**.
- Test đã từng được xem trong các lần chạy trước (khi quy trình cũ còn chọn model theo Test), nên không còn là tập chưa từng quan sát.
