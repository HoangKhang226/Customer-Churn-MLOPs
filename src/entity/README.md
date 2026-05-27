# 📋 Entity - Thư mục định nghĩa Data Entities

## 📁 Cấu trúc thư mục

```
src/entity/
└── config_entity.py        # Dataclasses cho config objects
```

---

## 📄 Mô tả file

### `config_entity.py`

**Chức năng chính**: Định nghĩa các dataclass để lưu trữ configuration cho từng stage

**Tại sao dùng dataclass?**
- **Type-safe**: Kiểm tra kiểu dữ liệu tại compile time, tránh lỗi runtime
- **Immutable**: `frozen=True` ngăn chặn thay đổi config sau khi khởi tạo
- **Clean code**: Dễ đọc hơn dict, có autocomplete trong IDE

---

## 📦 Các Dataclass được định nghĩa

### 1. `DataIngestionConfig`
**Cho Stage 1**: Data Ingestion

**Attributes**:
- `root_dir`: Thư mục gốc cho stage này
- `local_data_file`: Đường dẫn đến file zip
- `unzip_dir`: Thư mục đích để giải nén

---

### 2. `DataValidationConfig`
**Cho Stage 2**: Data Validation

**Attributes**:
- `root_dir`: Thư mục gốc cho stage này
- `STATUS_FILE`: Đường dẫn file status.txt (kết quả validation)
- `unzip_data_dir`: Đường dẫn đến dữ liệu cần validate
- `all_schema`: Dictionary chứa schema từ schema.yaml

---

### 3. `DataTransformationConfig`
**Cho Stage 3**: Data Transformation

**Attributes**:
- `root_dir`: Thư mục gốc cho stage này
- `train_data_path`: Đường dẫn train.csv
- `test_data_path`: Đường dẫn test.csv
- `preprocessor_path`: Đường dẫn lưu preprocessor.joblib

---

### 4. `ModelTrainerConfig`
**Cho Stage 4**: Model Training

**Attributes**:
- `root_dir`: Thư mục gốc cho stage này
- `train_data_path`: Đường dẫn train_transformed.npz
- `test_data_path`: Đường dẫn test_transformed.npz
- `model_name`: Tên file model (model.joblib)
- `lgbm_params`: Dictionary chứa hyperparameters cho LightGBM
- `xgboost_params`: Dictionary chứa hyperparameters cho XGBoost
- `mlflow_uri`: MLflow tracking URI

---

### 5. `ModelEvaluationConfig`
**Cho Stage 5**: Model Evaluation

**Attributes**:
- `root_dir`: Thư mục gốc cho stage này
- `test_data_path`: Đường dẫn test_transformed.npz
- `model_path`: Đường dẫn model.joblib
- `all_params`: Dictionary chứa toàn bộ params từ params.yaml
- `metric_file_name`: Đường dẫn metrics.json
- `mlflow_uri`: MLflow tracking URI

---

## 🔄 Cách sử dụng

**Luồng hoạt động**:
1. ConfigurationManager đọc YAML files
2. Tạo config object (dataclass) tương ứng với stage
3. Truyền config object vào Component
4. Component truy cập config thông qua `self.config.attribute`

**Ví dụ**: 
- Component nhận: `DataIngestionConfig` object
- Truy cập: `self.config.root_dir`, `self.config.local_data_file`
- Thay vì: `config["root_dir"]`, `config["local_data_file"]` (dict)
