# Nhận xét dữ liệu — Pima Indians Diabetes

Dữ liệu: file `pima-indians-diabetes.csv` tải từ Kaggle (https://www.kaggle.com/datasets/kumargh/pimaindiansdiabetescsv) ngày 10/11/2026. Gồm 768 mẫu, 8 feature số và nhãn `Outcome`. Theo mô tả trên trang, nhãn cho biết bệnh nhân có khởi phát tiểu đường hay không. 

## 1. Phân bố các biến
- Nhãn lệch vừa phải: 500 mẫu lớp 0 (65,1%) và 268 mẫu lớp 1 (34,9%). Vì vậy khi đánh giá mô hình không nên chỉ dùng Accuracy.
- Không có giá trị NaN và không có dòng trùng.
- Insulin (độ lệch 2,27), DiabetesPedigreeFunction (1,92), Age (1,13) và Pregnancies (0,90) lệch phải rõ: trung bình lớn hơn trung vị, ví dụ Age có trung bình 33,2 và trung vị 29.
- BloodPressure lệch trái (−1,84) vì có 35 giá trị bằng 0.
- Glucose, SkinThickness và BMI gần đối xứng.

## 2. Giá trị bất thường
- Dữ liệu thiếu được ghi bằng số 0 chứ không phải NaN. Số 0 ở Insulin: 374 (48,7%), SkinThickness: 227 (29,6%), BloodPressure: 35 (4,6%), BMI: 11 (1,4%), Glucose: 5 (0,7%).
- 376 dòng (49,0%) có ít nhất một số 0 ở năm biến này. Cả 227 dòng có SkinThickness = 0 đều có Insulin = 0, nên hai biến thiếu cùng nhau.
- Glucose, BloodPressure và BMI bằng 0 không thể là giá trị thật. Với Insulin và SkinThickness thì nhiều khả năng cũng là thiếu, nhưng nguồn không giải thích nên đây mới là suy đoán.
- Pregnancies = 0 (111 mẫu) là giá trị hợp lệ, không coi là thiếu.
- Tỷ lệ số 0 khác nhau giữa hai lớp (Insulin 47,2% ở lớp 0 và 51,5% ở lớp 1), nên việc thiếu có thể không ngẫu nhiên và cách xử lý có thể ảnh hưởng kết quả.
- Trung vị Insulin của lớp 1 bằng 0, nên so sánh Insulin giữa hai lớp trên dữ liệu thô dễ gây hiểu sai.
- Có vài giá trị rất lớn (Insulin tối đa 846, SkinThickness tối đa 99, Pregnancies tối đa 17). Chưa có bằng chứng đây là lỗi nhập nên giữ nguyên.

## 3. Mối liên hệ với nhãn
- Glucose liên hệ mạnh nhất với Outcome (hệ số Pearson 0,467). Glucose trung bình là 110,0 ở lớp 0 và 141,3 ở lớp 1. Tỷ lệ lớp 1 là 7,3% ở nhóm Glucose ≤ 99, 27,7% ở nhóm 100–125 và 59,3% ở nhóm > 125 (đã bỏ các số 0).
- BMI (0,293), Age (0,238) và Pregnancies (0,222) liên hệ ở mức trung bình thấp; DiabetesPedigreeFunction thấp hơn (0,174).
- Insulin, SkinThickness và BloodPressure chỉ có hệ số 0,065–0,131 trên dữ liệu thô, phần lớn do số 0 gây nhiễu. Khi coi số 0 là thiếu, hệ số tăng lên (Insulin 0,303; SkinThickness 0,259; BloodPressure 0,171). Vì vậy không nên kết luận ba biến này ít thông tin.
- Quan hệ không hoàn toàn tuyến tính. Tỷ lệ lớp 1 theo nhóm tuổi là 21,6% (21–30), 48,4% (31–40), 56,6% (41–50) và 46,9% (trên 50, chỉ 81 mẫu nên chưa đủ cơ sở nêu xu hướng).
- Một số feature tương quan với nhau (Pregnancies và Age: 0,54; BMI và SkinThickness: 0,65 sau khi coi 0 là thiếu), cần lưu ý khi diễn giải mô hình tuyến tính.

## 4. Giới hạn của dữ liệu
1. Chỉ có 768 mẫu, trong đó lớp dương chỉ 268 mẫu, nên các chỉ số đánh giá có độ bất định lớn.
2. Trang nguồn chỉ nói đây là hồ sơ y tế của người Pima Indians, không nêu giới tính hay vùng lấy mẫu (trong dữ liệu, tuổi nhỏ nhất là 21). Kết quả không nên khái quát sang nhóm khác.
3. Dữ liệu thiếu mã hóa bằng số 0, chiếm tỷ lệ lớn ở Insulin và SkinThickness, và có thể không ngẫu nhiên.
4. Nhiều biến lệch phải; nhóm tuổi trên 50 có ít mẫu.
5. Đây là bản đăng lại trên Kaggle; trang không nêu ai thu thập, cách chẩn đoán hay ý nghĩa từng cột. Nguồn gốc dữ liệu chưa được kiểm tra.
6. Tương quan và tỷ lệ theo nhóm chỉ là mô tả. Dữ liệu quan sát không cho biết biến nào gây ra bệnh, và chưa kiểm soát nhiễu (ví dụ Age và Pregnancies tương quan với nhau).
