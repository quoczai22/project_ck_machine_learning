# Thử nghiệm LDA và khám phá KMeans (Chương 3 và 4)

**Phạm vi:** chỉ dùng tập Train (614 mẫu); không đọc Test. Tiền xử lý (median, Glucose/BMI, scaler) nằm trong Pipeline hoặc fit trên Train. Mô hình đã chốt: SVM RBF (C=0,1, balanced, gamma='scale').

## 1. LDA (Chương 4 — Giảm chiều dữ liệu)

5-fold StratifiedKFold trên Train, mean ± std (`bang_lda_ket_qua.csv`). LDA cho bài toán nhị phân chỉ có tối đa 1 thành phần.

| Phương án | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| SVM đã chốt, không giảm chiều | 0,7947 ± 0,0607 | 0,6699 | 0,8298 | 0,7285 |
| LDA 1 thành phần + SVM | 0,7150 ± 0,0450 | 0,6722 | 0,8235 | 0,6865 |
| LDA phân lớp trực tiếp (priors mặc định) | 0,5841 ± 0,0184 | 0,6617 | 0,8432 | 0,7443 |
| LDA phân lớp trực tiếp (priors 0,5/0,5) | 0,6869 ± 0,0233 | 0,6714 | 0,8432 | 0,7443 |

- Theo quy tắc Recall-first (F1 và ROC-AUC trong 0,03), phương án được chọn vẫn là **SVM đã chốt**. LDA không làm tăng Recall.
- LDA phân lớp trực tiếp có ROC-AUC và PR-AUC **cao hơn** SVM (0,8432 so với 0,8298; 0,7443 so với 0,7285), nghĩa là xếp hạng các ca dương tốt hơn, nhưng Recall tại ngưỡng mặc định thấp hơn (đặt priors cân bằng làm Recall tăng từ 0,584 lên 0,687). Chênh lệch AUC nhỏ hơn hoặc xấp xỉ độ lệch chuẩn nên chưa đủ để kết luận.
- LDA chỉ có 1 thành phần cho bài toán nhị phân, nên giữ lại rất ít thông tin so với PCA nhiều thành phần.

## 2. KMeans (Chương 3 — Học không giám sát)

Phân cụm không dùng nhãn trên 614 mẫu Train đã chuẩn hóa, k = 2–8 (`bang_kmeans_do_do.csv`, `bang_kmeans_mo_ta_cum.csv`, `kmeans_elbow_silhouette.png`). Nhãn Outcome chỉ dùng sau khi phân cụm để mô tả.

| k | Silhouette | ARI với Outcome |
|---|---|---|
| 2 | 0,1994 | 0,2222 |
| 3 | 0,1549 | 0,0966 |
| 4 | 0,1703 | 0,1009 |
| 5 | 0,1650 | 0,0871 |
| 8 | 0,1442 | 0,0616 |

- Silhouette cao nhất ở k = 2 nhưng vẫn **thấp (0,1994)**, cho thấy cấu trúc cụm trong dữ liệu yếu; đường Elbow không có điểm gãy rõ.
- Với k = 2: cụm 0 (245 mẫu) có tỷ lệ Outcome = 1 là **60,8%**, Glucose trung bình 145,5, tuổi trung bình 42,3; cụm 1 (369 mẫu) có tỷ lệ **17,6%**, Glucose 105,8, tuổi 27,4. Hai cụm phân biệt nhóm nguy cơ cao với nguy cơ thấp, chủ yếu theo Glucose và tuổi.
- ARI với Outcome chỉ 0,22 (rất thấp), nên phân cụm không thay thế được mô hình phân lớp có giám sát.
- Đây là phân tích khám phá; không dùng cụm làm feature và không ảnh hưởng mô hình cuối.

## 3. Quyết định

Mô hình cuối giữ nguyên (SVM, không giảm chiều). Các thử nghiệm PCA, LDA và KMeans là phần vận dụng Chương 3 và 4 của đề cương. Chương 5 (Học tăng cường) không áp dụng được vì dữ liệu Pima là bảng cố định, không có môi trường, trạng thái hay chuỗi hành động.
