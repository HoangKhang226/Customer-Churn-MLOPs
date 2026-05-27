# ⚙️ Config - Thư mục quản lý cấu hình

## 📁 Cấu trúc thư mục

```
src/config/
├── __init__.py
└── configuration.py        # ConfigurationManager class
```

---

## 📄 Mô tả từng file

### 1. `__init__.py`

**Chức năng**: File đánh dấu thư mục này là Python package, cho phép import các module bên trong

---

### 2. `configuration.py`

**Chức năng chính**: Quản lý tập trung tất cả cấu hình của project

**Class**: `ConfigurationManager`

**Làm gì**:
- **Đọc file YAML**: 
  - `config.yaml`: Đường dẫn artifacts cho từng stage
  - `schema.yaml`: Schema validation cho dữ liệu
  - `params.yaml`: Hyperparameters cho model training
- **Tạo thư mục**: Tự động tạo thư mục `artifacts/` nếu chưa tồn tại
- **Cung cấp config objects**: Trả về config object (dataclass) cho từng stage

**Methods chính**:
- `get_data_ingestion_config()` → `DataIngestionConfig`
  - Trả về: root_dir, local_data_file, unzip_dir
  
- `get_data_validation_config()` → `DataValidationConfig`
  - Trả về: root_dir, STATUS_FILE, unzip_data_dir, all_schema
  
- `get_data_transformation_config()` → `DataTransformationConfig`
  - Trả về: root_dir, train_data_path, test_data_path, preprocessor_path
  
- `get_model_trainer_config()` → `ModelTrainerConfig`
  - Trả về: root_dir, train_data_path, test_data_path, model_name, lgbm_params, xgboost_params, mlflow_uri
  
- `get_model_evaluation_config()` → `ModelEvaluationConfig`
  - Trả về: root_dir, test_data_path, model_path, all_params, metric_file_name, mlflow_uri

**Lợi ích**:
- **Centralized**: Tất cả config ở một nơi, dễ quản lý
- **Type-safe**: Trả về dataclass objects với type hints
- **Flexible**: Chỉ cần sửa YAML files, không cần sửa code
- **Reusable**: Có thể tái sử dụng cho nhiều runs

**Cách hoạt động**:
1. Khởi tạo ConfigurationManager (đọc 3 file YAML)
2. Gọi method `get_<stage>_config()` tương ứng
3. Nhận config object (dataclass) để truyền vào component
