# Biên bản rà soát tổng hợp — DA 02: Pima Indians Diabetes

**Việc 20** (Điều phối và kiểm tra đủ deliverable) — người phụ trách: Trịnh Hữu Kiến Quốc.

Phạm vi: đối chiếu các việc trong PCCV (bản local cập nhật 02/10; cần đối chiếu với bản SharePoint) với bằng chứng trên GitHub (`quoczai22/project_ck_machine_learning`). Chỉ ghi **hoàn thành** khi có bằng chứng cụ thể.

| Ký hiệu | Ý nghĩa |
|---|---|
| ✅ HT | Hoàn thành, có bằng chứng |
| 🟡 HTC | Hoàn thành có điều kiện (còn chữ ký, merge hoặc ghi chú) |
| 🟠 DD | Đang làm, còn thiếu |
| ⬜ CH | Chưa làm |

---

## 1. Sỹ Luân — Dữ liệu và EDA (việc 1–4)

| Việc | Nội dung | Trạng thái | Ghi chú |
|---|---|---|---|
| 1 | Nguồn, SHA-256, ngày tải | 🟠 DD | Nội dung file nhánh `Luan` giống hệt file đang dùng (768×9, không khác giá trị). SHA-256 khác do ký tự xuống dòng (CRLF/LF). Nguồn Kaggle chưa xác nhận, chưa có ngày tải. |
| 2 | Bảng kiểm tra chất lượng | ✅ HT | `data_quality_check.csv`: 768 dòng, 500/268, không thiếu, không trùng |
| 3 | Histogram, boxplot | 🟡 HTC | Có trong `Luan/…/images/eda_raw/`, chưa merge vào nhánh chung |
| 4 | Heatmap và nhận xét | 🟠 DD | Có heatmap; nhận xét còn ngắn, cần có giới hạn, không suy diễn nhân quả |

**Còn thiếu:** xác nhận nguồn Kaggle (ghi URL, ngày tải, SHA-256 của chính bản đó); merge nhánh `Luan`.

---

## 2. Nguyễn Tấn Kỳ — Tiền xử lý, split, leakage (việc 5–8)

| Việc | Nội dung | Trạng thái | Ghi chú |
|---|---|---|---|
| 5 | Zero-as-missing, bảng quyết định | ✅ HT | `viet5_bang_quyet_dinh.csv`: đúng 5 cột, Pregnancies giữ nguyên |
| 6 | Pipeline tiền xử lý | 🟡 HTC | Đúng (0→NaN, median, scaler). Đã bỏ đánh giá Test (PR #3 đã merge). Pipeline dùng Random Forest, **không phải model cuối** |
| 7 | Split 80/20 stratify | ✅ HT | Train 614 (214 dương), Test 154 (54 dương), `random_state=42` |
| 8 | Test leakage | 🟠 DD | Hai script audit PASS khi chạy trên máy (log trong `docs/logs/`). **Thiếu** log đủ 6 bài test (`Viec_8/code.txt`, Phần 7); hiện chỉ có 3 kiểm tra |

**Còn thiếu:** chạy lại trên Colab và lưu log vào repo; đồng bộ cột đích `Class` với `Outcome`.

---

## 3. Đoàn Bảo — Feature engineering (việc 9–12)

| Việc | Nội dung | Trạng thái | Ghi chú |
|---|---|---|---|
| 9 | Baseline 8 feature | 🟡 HTC | Có CV trên Train với Recall, F1, ROC-AUC, PR-AUC. Cần kiểm tra lại sau PR |
| 10 | Thử nghiệm feature | 🟡 HTC | Bảng có F1, ROC-AUC, PR-AUC. Cần thống nhất với bộ 9 feature đã chốt |
| 11 | Chọn lọc feature | 🟡 HTC | Đã chọn theo quy tắc Recall-first (PR #5 đã merge). Kết quả: "Không chọn" (Recall 0,6123). **`feature_pipeline_for_app.pkl` vẫn là Logistic Regression**, không phải model cuối |
| 12 | Báo cáo feature | 🟠 DD | Cập nhật theo F1, ROC-AUC, PR-AUC và quy tắc đã chốt. Ngưỡng BMI/tuổi phải khớp với bộ đã chốt. Không nêu "tối ưu" khi chênh lệch nằm trong độ lệch chuẩn |

---

## 4. Nguyễn Hiệu — Baseline, GridSearch, chọn model (việc 13–16)

| Việc | Nội dung | Trạng thái | Ghi chú |
|---|---|---|---|
| 13 | Baseline LR, RF, SVM | ✅ HT | CV trên Train, đủ Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC |
| 14 | GridSearch | 🟡 HTC | Đã sửa `refit` sang Recall và cố định scenario 2 (PR #2, #4 đã merge). **Cần chạy lại** và quyết định phần đánh giá Test từng model |
| 15 | Chọn candidate | ✅ HT | SVM, scenario 2, C=0,1, balanced, gamma=scale. Ngưỡng 0,03 đã chốt qua tin nhắn ngày 09/11/2026 (Hiệu và Kiến Quốc). Chưa có bản chữ ký |
| 16 | Lý thuyết và ví dụ LR, RF, SVM | ⬜ CH | Chưa có trong repo. Hiệu phải tự viết |

---

## 5. Phạm Tùng — Artifact, app, demo (việc 21–24)

| Việc | Nội dung | Trạng thái | Ghi chú |
|---|---|---|---|
| 21 | Artifact và metadata | 🟡 HTC | Có `.joblib` và `run_metadata.json` (có `feature_order`, `final_params`). `feature_order` do Kiến Quốc bổ sung, Tùng cần xác nhận |
| 22 | Streamlit | 🟡 HTC | App chạy; đã kiểm thử bằng `streamlit.testing`. Thông báo khi thiếu model do Kiến Quốc sửa, Tùng cần xác nhận |
| 23 | Kiểm thử app | 🟡 HTC | Kiến Quốc đã chạy các trường hợp: mặc định, toàn 0, giá trị lớn, thiếu model. **Tùng cần lập checklist chính thức** và chạy trên trình duyệt |
| 24 | README, slide, demo | 🟠 DD | README có phần ứng dụng. Chưa có slide và demo script |

---

## 6. Trịnh Hữu Kiến Quốc — Đánh giá và điều phối (việc 17–20)

| Việc | Nội dung | Trạng thái | Ghi chú |
|---|---|---|---|
| 17 | Protocol metric | ✅ HT | Đã có trong code, README, metadata. Đã chốt qua tin nhắn ngày 09/11/2026 (Hiệu và Kiến Quốc) |
| 18 | Đánh giá candidate trên Test | 🟡 HTC | Có Recall, F1, ROC-AUC, PR-AUC, Accuracy, confusion matrix. **Ghi chú bắt buộc:** Test đã được xem nhiều lần, không phải lần đầu |
| 19 | Phân tích FN/FP | 🟡 HTC | `reports/error_analysis.md`. **Cần Bảo rà soát** cách diễn giải |
| 20 | Điều phối, rà soát | 🟠 DD | Biên bản này là bước đầu. Còn thu phản hồi từng thành viên và cập nhật PCCV |

---

## 7. Vấn đề chung cần xử lý trước khi nộp

1. **Tập Test đã được xem nhiều lần.** Báo cáo phải liệt kê đầy đủ các cấu hình đã tính trên Test.
2. **Repo:** thư mục DA 02 chưa có git riêng (đang nằm trong repo cha không liên quan); phần đã chốt chưa được commit.
3. **Báo cáo (12 mục) và slide:** chưa có, phải theo cấu trúc đề yêu cầu.
4. **Cổng 11 và 12:** mỗi người viết và hiểu phần mình; vấn đáp trực tiếp.
5. **PR #1 (Bảo → main) vẫn mở** và chứa commit của Kỳ. Cần tách trước khi merge.
6. **PCCV:** cập nhật trạng thái trên SharePoint sau khi nhóm thống nhất.

---

## 8. Xác nhận

Biên bản được xác nhận qua tin nhắn; không yêu cầu chữ ký.

| Thành viên | Trạng thái xác nhận |
|---|---|
| Trịnh Hữu Kiến Quốc (điều phối) | Đã xác nhận |
| Nguyễn Hiệu | Đã xác nhận việc 15, 17 (ngày 09/11/2026) |
| Đoàn Bảo | Chưa xác nhận |
| Nguyễn Tấn Kỳ | Chưa xác nhận |
| Sỹ Luân | Chưa xác nhận |
| Phạm Tùng | Chưa xác nhận |
