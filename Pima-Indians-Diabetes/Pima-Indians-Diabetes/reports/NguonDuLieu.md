# Nguồn dữ liệu và kiểm chứng

## 1. Thông tin nguồn
| Mục | Giá trị |
|---|---|
| Tên dataset trên Kaggle | pima-indians-diabetes.csv |
| URL | https://www.kaggle.com/datasets/kumargh/pimaindiansdiabetescsv |
| Người đăng | Kumar (kumargh), Version 1, cập nhật lần cuối "9 years ago" tính đến 10/11/2026 |
| License ghi trên trang | CC0: Public Domain |
| File | pima-indians-diabetes.csv (24,05 kB trên trang; 24.045 byte file đã tải), 9 cột |
| Ngày tải | 10/11/2026 |
| SHA-256 file tải về (nguyên bản) | `c1e0f7d59d2e366cff1ab9fee1796e1e381ea5da64c6d96ef4baabf1eda5f238` |

## 0. Ghi chú nguồn thay thế (bổ sung)

Đề tài DA 02 chỉ định nguồn `https://www.kaggle.com/uciml/pima-indians-diabetes-database`. Trang này không truy cập được tại thời điểm tải dữ liệu, nên nhóm dùng bản `kumargh/pimaindiansdiabetescsv` làm nguồn thay thế.

- Trang `uciml` không truy cập được: ngày thử truy cập `[điền ngày]`, số lần thử `[điền]`.
- Nội dung bản `kumargh` được đối chiếu với file đang dùng trong pipeline (`data/raw/diabetes.csv`): 768 dòng × 9 cột, giá trị từng ô giống nhau, khác nhau chỉ ở ký tự xuống dòng (CRLF/LF). Không có dòng tiêu đề trong cả hai file.
- Chưa đối chiếu được với bản `uciml` vì trang không truy cập được. Nếu sau này truy cập được, cần tải lại và so sánh.
- Việc thay thế nguồn đã được báo cho `[người/giảng viên đã báo]` vào ngày `[điền ngày]`.
