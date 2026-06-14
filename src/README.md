## 📂 Chi tiết các thư mục

### 1️⃣ `config/` - Thư mục cấu hình

```
config/
├── config.yaml          # Cấu hình đường dẫn artifacts cho từng stage
├── schema.yaml          # Schema validation cho dữ liệu
├── logging.yaml         # Cấu hình logging (format, handlers, levels)
└── params.yaml          # Không gian tìm kiếm hyperparameter cho Optuna
```

#### 📄 Giải thích từng file:

- **`config.yaml`**:
  - Định nghĩa `artifacts_root` và đường dẫn cho từng stage
  - Cấu hình MLflow URI cho tracking
  - Ví dụ: `data_ingestion.root_dir`, `model_trainer.train_data_path`

- **`schema.yaml`**:
  - Định nghĩa kiểu dữ liệu cho từng cột (int64, float64, object)
  - Chỉ định cột target (`Churn`)
  - Dùng để validate dữ liệu trong Stage 2

- **`logging.yaml`**:
  - Cấu hình format log (timestamp, level, message)
  - Định nghĩa handlers (console, file)
  - Thiết lập log levels cho từng module

- **`params.yaml`**:
  - Hyperparameters cho LightGBM: `n_estimators`, `max_depth`, `learning_rate`
  - Hyperparameters cho XGBoost: `n_estimators`, `max_depth`, `learning_rate`
  - Cấu hình vùng không gian tìm kiếm (search space) cho Optuna (n_trials=10)

---

### 2️⃣ `src/components/` - Components xử lý logic

```
src/components/
├── data_ingestion.py           # Giải nén dữ liệu từ zip
├── data_validation.py          # Kiểm tra schema của dữ liệu
├── data_transformation.py      # Feature engineering & preprocessing
├── model_trainer.py            # Training & Tuning với Optuna
├── model_evaluation.py         # Evaluation và visualization
└── prediction.py               # Dự đoán dữ liệu mới (Inference)
```

#### 📄 Giải thích từng file:

- **`data_ingestion.py`** (Stage 1):
  - **Class**: `DataIngestion`
  - **Chức năng**: Giải nén file `playground-series-s6e3.zip` từ thư mục `data/`
  - **Output**: `train.csv`, `test.csv` trong `artifacts/data_ingestion/`
  - **Method chính**: `extract_zip_file()`

- **`data_validation.py`** (Stage 2):
  - **Class**: `DataValidation`
  - **Chức năng**: Kiểm tra xem tất cả các cột trong dữ liệu có khớp với `schema.yaml` không
  - **Output**: `status.txt` (True/False) trong `artifacts/data_validation/`
  - **Method chính**: `validate_all_columns()`

- **`data_transformation.py`** (Stage 3):
  - **Class**: `DataTransformation`, `ChurnFeatureEngineer`, `WinsorizerTransformer`
  - **Chức năng**:
    - Tạo 9 features mới từ EDA insights
    - Preprocessing: Imputation, Scaling, Encoding
    - Áp dụng SMOTE để cân bằng nhãn (50:50)
  - **Output**:
    - `train_transformed.npz` (920,754 samples, 23 features)
    - `test_transformed.npz` (254,655 samples, 23 features)
    - `preprocessor.joblib` (sklearn pipeline đã fit)
  - **Method chính**: `initiate_data_transformation()`

- **`model_trainer.py`** (Stage 4):
  - **Class**: `ModelTrainer`
  - **Chức năng**:
    - Load dữ liệu từ `.npz` files
    - Chia train/validation (80/20)
    - Tuning siêu tham số với Optuna cho LightGBM, XGBoost, CatBoost (15 trials)
    - Xây dựng mô hình Stacking (Ensemble) kết hợp cả 3 mô hình trên
    - So sánh và chọn mô hình tốt nhất dựa trên ROC AUC
    - Log tất cả vào MLflow
  - **Output**:
    - `model.joblib` (mô hình tốt nhất)
    - `metrics.json` (accuracy, precision, recall, f1, roc_auc)
  - **Method chính**: `initiate_model_trainer()`

- **`model_evaluation.py`** (Stage 5):
  - **Class**: `ModelEvaluation`
  - **Chức năng**:
    - Load mô hình đã train
    - Tạo predictions trên test set
    - Tính toán metrics (nếu có nhãn)
    - Tạo visualizations: Confusion Matrix, ROC Curve
    - Log artifacts vào MLflow
  - **Output**:
    - `predictions.npz` (predictions cho test set)
    - `metrics.json` (nếu có nhãn test)
    - `confusion_matrix.png`, `roc_curve.png` (nếu có nhãn test)
  - **Method chính**: `initiate_model_evaluation()`

- **`prediction.py`** (Stage 6):
  - **Class**: `PredictionPipeline`
  - **Chức năng**:
    - Load mô hình tốt nhất đã train và pipeline preprocessor
    - Load dữ liệu test chưa gán nhãn
    - Thực hiện tiền xử lý và dự đoán xác suất rời bỏ (Churn probabilities)
    - Tạo file output phục vụ submission
  - **Output**: `submission.csv`
  - **Method chính**: `predict()`

---

### 3️⃣ `src/config/` - Configuration Management

```
src/config/
├── __init__.py
└── configuration.py        # ConfigurationManager class
```

#### 📄 Giải thích:

- **`configuration.py`**:
  - **Class**: `ConfigurationManager`
  - **Chức năng**:
    - Đọc các file YAML (`config.yaml`, `schema.yaml`, `params.yaml`)
    - Tạo thư mục artifacts nếu chưa tồn tại
    - Cung cấp methods để lấy config cho từng stage
  - **Methods**:
    - `get_data_ingestion_config()` → `DataIngestionConfig`
    - `get_data_validation_config()` → `DataValidationConfig`
    - `get_data_transformation_config()` → `DataTransformationConfig`
    - `get_model_trainer_config()` → `ModelTrainerConfig`
    - `get_model_evaluation_config()` → `ModelEvaluationConfig`
    - `get_prediction_config()` → `PredictionConfig`

---

### 4️⃣ `src/entity/` - Data Entities

```
src/entity/
└── config_entity.py        # Dataclasses cho config objects
```

#### 📄 Giải thích:

- **`config_entity.py`**:
  - **Dataclasses** (frozen=True để immutable):
    - `DataIngestionConfig`: root_dir, local_data_file, unzip_dir
    - `DataValidationConfig`: root_dir, STATUS_FILE, unzip_data_dir, all_schema
    - `DataTransformationConfig`: root_dir, train_data_path, test_data_path, preprocessor_path
    - `ModelTrainerConfig`: root_dir, train_data_path, test_data_path, model_name, lgbm_params, xgboost_params, mlflow_uri
    - `ModelEvaluationConfig`: root_dir, test_data_path, model_path, all_params, metric_file_name, mlflow_uri
    - `PredictionConfig`: model_path, preprocessor_path, test_data_path, submission_output_path
  - **Mục đích**: Type-safe configuration objects, dễ dàng truyền giữa các components

---

### 5️⃣ `src/pipeline/` - Pipeline Wrappers

```
src/pipeline/
├── __init__.py
├── stage_01_data_ingestion.py          # Wrapper cho Stage 1
├── stage_02_data_validation.py         # Wrapper cho Stage 2
├── stage_03_data_transformation.py     # Wrapper cho Stage 3
├── stage_04_model_trainer.py           # Wrapper cho Stage 4
├── stage_05_model_evaluation.py        # Wrapper cho Stage 5
└── stage_06_prediction.py              # Wrapper cho Stage 6
```

#### 📄 Giải thích từng file:

Mỗi file pipeline có cấu trúc tương tự:

- **Class**: `<StageName>TrainingPipeline`
- **Method**: `main()`
  1. Khởi tạo `ConfigurationManager`
  2. Lấy config cho stage tương ứng
  3. Khởi tạo component với config
  4. Gọi method chính của component

**Ví dụ**: `stage_04_model_trainer.py`

```python
class ModelTrainerTrainingPipeline:
    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer_config)
        model_trainer.initiate_model_trainer()
```

**Mục đích**:

- Tách biệt logic component và orchestration
- Dễ dàng chạy từng stage độc lập để debug
- Có thể chạy: `python src/pipeline/stage_04_model_trainer.py`

---

### 6️⃣ `src/utils/` - Utility Functions

```
src/utils/
├── common.py           # Các hàm tiện ích chung
└── logger.py           # Logger configuration
```

#### 📄 Giải thích:

- **`common.py`**:
  - `read_yaml()`: Đọc file YAML và trả về ConfigBox
  - `create_directories()`: Tạo danh sách thư mục
  - `save_json()`, `load_json()`: Lưu/đọc JSON
  - `save_bin()`, `load_bin()`: Lưu/đọc binary files (joblib)
  - `get_size()`: Lấy kích thước file

- **`logger.py`**:
  - Setup logging từ `config/logging.yaml`
  - Tạo logger instance dùng chung: `logger`
  - Tự động tạo thư mục `logs/` nếu chưa có
  - Cấu hình UTF-8 encoding cho Windows
