# DA 02 – Pima Indians Diabetes: Việc 21–24

Bộ triển khai này được tạo từ mã nguồn và artifact có sẵn trong ZIP nhóm cung cấp. **Không huấn luyện lại mô hình.** Ứng dụng nạp `artifacts/pipeline.joblib`, tạo đúng feature của scenario `2_Them_Ratio`, sau đó gọi `predict()` và `predict_proba()`.

## 1. Mô hình đã chốt

- Scenario: `2_Them_Ratio` (8 feature gốc + `Glucose_BMI_Ratio`).
- Mô hình: SVM RBF, `C=0.1`, `class_weight=balanced`, `gamma=scale`, `probability=True`, `random_state=42`.
- Lựa chọn trên Train bằng StratifiedKFold 5-fold + GridSearchCV; Recall là metric ưu tiên, có ràng buộc F1 và ROC-AUC trong sai số 0,03.
- Pipeline đã fit: `SimpleImputer(strategy="median") → StandardScaler() → SVC(RBF)`.
- Tiền xử lý dữ liệu đầu vào: 0 ở Glucose, BloodPressure, SkinThickness, Insulin, BMI được đổi thành NaN; Pregnancies giữ nguyên. Tạo `Glucose_BMI_Ratio = Glucose / BMI`.
- Nhãn cuối lấy từ `model.predict()`. Xác suất lớp 1 chỉ hiển thị bổ sung; không áp ngưỡng 0,5 thủ công để thay đổi nhãn.

### Metric được ghi trong hồ sơ nhóm

| Metric | CV trên Train | Test sau khi chốt |
|---|---:|---:|
| Accuracy | 0.7296 | 0.6688 |
| Precision | 0.5835 | 0.5190 |
| Recall | 0.7947 | 0.7593 |
| F1 | 0.6713 | 0.6165 |
| ROC-AUC | 0.8303 | 0.7831 |

Lưu ý: biên bản review nhóm cho biết Test đã được xem nhiều lần. Vì vậy, không mô tả đây là lần đánh giá Test đầu tiên hoặc dùng lại Test để chọn mô hình.

## 2. Cài đặt và chạy

Khuyến nghị Python 3.11 hoặc phiên bản tương thích với các wheel của scikit-learn 1.9.0.

```bash
cd Deployment_Viec21_24
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Mở địa chỉ Local URL mà Streamlit in ra (thường là `http://localhost:8501`).

## 3. Kiểm thử

```bash
pytest -q
```

Checklist thao tác trình duyệt nằm ở `docs/TEST_CHECKLIST.md`.

## 4. Cấu trúc

```text
Deployment_Viec21_24/
├── app.py
├── prediction.py
├── test_app.py
├── requirements.txt
├── README.md
├── artifacts/
│   ├── pipeline.joblib
│   └── metadata.json
└── docs/
    ├── TEST_CHECKLIST.md
    ├── DEMO_SCRIPT.md
    └── SLIDE_OUTLINE.md
```

## 5. Lưu ý tương thích và an toàn

Artifact gốc được lưu bằng scikit-learn 1.9.0. Không nên nạp bằng phiên bản khác; hãy cài đúng phiên bản đã ghim trong `requirements.txt`. Nếu môi trường của bạn không cung cấp wheel tương thích, cần chạy trong môi trường Python được scikit-learn 1.9.0 hỗ trợ thay vì tự ý thay đổi artifact.

Đây là sản phẩm học tập trên bộ dữ liệu Pima, không phải công cụ chẩn đoán. Không nhập dữ liệu định danh cá nhân hoặc sử dụng kết quả để quyết định điều trị.
