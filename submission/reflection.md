# BÁO CÁO KẾT QUẢ DỰ ÁN MLOPS PIPELINE

**Người thực hiện:** Lâm Hoàng Hải
**Dự án:** Wine Quality Prediction (Automated Pipeline)

## 1. Kết quả lựa chọn Siêu tham số (Bước 1)

Sau quá trình thực nghiệm cục bộ và theo dõi qua MLflow UI, bộ siêu tham số tối ưu nhất đã được lựa chọn để đưa vào sản xuất:

| Siêu tham số        | Giá trị | Lý do lựa chọn                                                                                                                  |
| --------------------- | --------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `n_estimators`      | 100       | Đảm bảo độ ổn định của mô hình mà không làm chậm quá trình huấn luyện quá mức.                                |
| `max_depth`         | 50        | Giúp mô hình học được các đặc trưng phức tạp của dữ liệu rượu nhưng vẫn tránh được Overfitting quá nặng. |
| `min_samples_split` | 3         | Tăng khả năng tổng quát hóa (Generalization) của mô hình trên tập dữ liệu đánh giá.                                |

**Kết quả:** Độ chính xác (Accuracy) đạt khoảng **0.66** ở Phase 1 và tăng lên trên **0.70** sau khi bổ sung dữ liệu ở Phase 2.

## 2. Các khó khăn gặp phải và Cách giải quyết

Trong quá trình triển khai hệ thống CI/CD từ local lên Cloud, tôi đã gặp một số thách thức kỹ thuật đáng chú ý:

### 2.1. Lỗi xác thực DVC (401 Invalid Credentials)

* **Vấn đề:** Pipeline bị dừng ở bước `dvc pull` do không thể truy cập Google Cloud Storage mặc dù đã cung cấp file JSON key.
* **Giải quyết:**
  * Sử dụng lệnh `dvc remote modify myremote --unset credentialpath` để hệ thống tự động nhận diện Key qua biến môi trường.
  * Cấp thêm các Role `Storage Admin` và `Compute Admin` cho Service Account trên GCP Console.
  * Chuyển sang dùng `google-github-actions/auth@v2` để ghi nhận Secret an toàn hơn.

### 2.2. Lỗi cấu hình MLflow trên môi trường CI/CD

* **Vấn đề:** MLflow báo lỗi `MissingConfigException` khi chạy Unit Test và Train trên máy ảo GitHub.
* **Giải quyết:** Cấu hình biến môi trường `MLFLOW_TRACKING_URI=sqlite:///mlflow.db` trong file `mlops.yml` và trong code `train.py` để đồng nhất dữ liệu vào một file Database duy nhất.

### 2.3. Lỗi Health Check khi Deploy

* **Vấn đề:** Job Deploy báo lỗi dù Server đã khởi động thành công trên VM (lỗi Curl Health Check).
* **Giải quyết:** Tăng thời gian chờ (`sleep`) từ 5 giây lên 15 giây trong file workflow để đảm bảo server đã sẵn sàng phục vụ sau khi tải mô hình từ GCS.
