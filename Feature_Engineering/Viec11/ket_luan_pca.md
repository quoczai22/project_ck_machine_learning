# Thử nghiệm PCA (Chương 4 — Giảm chiều dữ liệu)

**Phạm vi:** chỉ dùng tập Train (614 mẫu) và 5-fold StratifiedKFold; không đọc Test. Imputer, ratio Glucose/BMI, scaler và PCA nằm trong Pipeline. Mô hình là SVM RBF đã chốt (C=0,1, balanced, gamma='scale').

**Kết quả** (`bang_pca_ket_qua.csv`, `bang_pca_phuong_sai.csv`; mean ± std qua 5 fold):

| Phương án | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| Không PCA (9 feature) | 0,7947 ± 0,0607 | 0,6699 | 0,8298 | 0,7285 |
| PCA 3 thành phần | 0,8176 ± 0,0407 | 0,6775 | 0,8158 | 0,6705 |
| PCA 5 thành phần | 0,8133 ± 0,0409 | 0,6673 | 0,8119 | 0,6871 |
| PCA 7 thành phần (95% phương sai) | 0,7900 ± 0,0581 | 0,6663 | 0,8249 | 0,7197 |

3 thành phần giữ 64,0% phương sai; 6 thành phần giữ 89,9%; 7 thành phần giữ 95,1%.

**Theo quy tắc Recall-first (F1 và ROC-AUC trong 0,03 của mức tốt nhất, rồi Recall cao nhất)**, phương án được chọn là PCA 3 thành phần.

**Nhận xét có giới hạn:**
- Chênh lệch Recall giữa PCA 3 thành phần và không PCA (0,023) nhỏ hơn độ lệch chuẩn qua các fold (khoảng 0,04–0,06), nên chưa đủ căn cứ để nói PCA cải thiện.
- PCA 3 thành phần làm giảm ROC-AUC (0,8158 so với 0,8298) và PR-AUC (0,6705 so với 0,7285), tức khả năng xếp hạng ca dương kém hơn dù Recall tại ngưỡng mặc định cao hơn.
- PCA 8 và 9 thành phần cho kết quả gần như trùng với không PCA, đúng như kỳ vọng.
- **Quyết định:** đây là thử nghiệm bổ sung. Mô hình cuối của nhóm giữ nguyên (không PCA), vì mô hình đã được chốt bằng biên bản và Test đã được xem nhiều lần; đổi mô hình sau đó sẽ làm mất ý nghĩa của lần đánh giá Test.
- PCA làm mất khả năng giải thích từng feature, điều không có lợi cho ứng dụng y tế mô phỏng này.
