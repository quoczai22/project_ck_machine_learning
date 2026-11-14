# Log kiểm tra leakage (Việc 8)

- `train_test_audit_log.csv`: kiểm tra imputer và scaler chỉ fit trên Train, và `predict` trên Test không fit lại.
- `cross_validation_train_audit_log.csv`: kiểm tra imputer và scaler trong từng fold CV trên Train.

Nguồn: script `train_test_pipeline_audit.py` và `cv_fold_train_audit.py` của Trịnh Hữu Kiến Quốc, nhánh `feature/tienxuli` (commit c9c610a).
Chạy trên máy (không dùng Colab) trên `pima_indians_zero_as_missing_NaN.csv` (768 dòng, cột đích `Class`).
Kết quả: cả hai PASS. Test_Accuracy trong log chỉ để báo cáo, không dùng để chọn model.
Chưa bao gồm đủ 6 bài test trong `Tien_Xu_Li/Viec_8/code.txt` (Phần 7).
