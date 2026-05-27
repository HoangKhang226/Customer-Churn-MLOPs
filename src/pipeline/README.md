# 🔄 Pipeline - Thư mục Pipeline Wrappers

## 📁 Cấu trúc thư mục

```
src/pipeline/
├── __init__.py
├── stage_01_data_ingestion.py          # Wrapper cho Stage 1
├── stage_02_data_validation.py         # Wrapper cho Stage 2
├── stage_03_data_transformation.py     # Wrapper cho Stage 3
├── stage_04_model_trainer.py           # Wrapper cho Stage 4
└── stage_05_model_evaluation.py        # Wrapper cho Stage 5
```

---

## 📄 Mô tả từng file

### 1. `__init__.py`

**Chức năng**: File đánh dấu thư mục này là Python package

---

### 2. `stage_01_data_ingestion.py`

**Class**: `DataIngestionTrainingPipeline`

**Chức năng**: Orchestrate Stage 1 - Data Ingestion

**Làm gì**:
- Khởi tạo ConfigurationManager
- Lấy config cho data ingestion
- Khởi tạo DataIngestion component với config
- Gọi method `extract_zip_file()` để giải nén dữ liệu
- Xử lý lỗi và logging

**Có thể chạy độc lập**: `python src/pipeline/stage_01_data_ingestion.py`

---

### 3. `stage_02_data_validation.py`

**Class**: `DataValidationTrainingPipeline`

**Chức năng**: Orchestrate Stage 2 - Data Validation

**Làm gì**:
- Khởi tạo ConfigurationManager
- Lấy config cho data validation
- Khởi tạo DataValidation component với config
- Gọi method `validate_all_columns()` để kiểm tra schema
- Xử lý lỗi và logging

**Có thể chạy độc lập**: `python src/pipeline/stage_02_data_validation.py`

---

### 4. `stage_03_data_transformation.py`

**Class**: `DataTransformationTrainingPipeline`

**Chức năng**: Orchestrate Stage 3 - Data Transformation

**Làm gì**:
- Khởi tạo ConfigurationManager
- Lấy config cho data transformation
- Khởi tạo DataTransformation component với config
- Gọi method `initiate_data_transformation()` để xử lý dữ liệu
- Xử lý lỗi và logging

**Có thể chạy độc lập**: `python src/pipeline/stage_03_data_transformation.py`

**Thời gian chạy**: ~30 giây

---

### 5. `stage_04_model_trainer.py`

**Class**: `ModelTrainerTrainingPipeline`

**Chức năng**: Orchestrate Stage 4 - Model Training

**Làm gì**:
- Khởi tạo ConfigurationManager
- Lấy config cho model trainer (bao gồm hyperparameters từ params.yaml)
- Khởi tạo ModelTrainer component với config
- Gọi method `initiate_model_trainer()` để train models
- Xử lý lỗi và logging

**Có thể chạy độc lập**: `python src/pipeline/stage_04_model_trainer.py`

**Thời gian chạy**: ~10-15 phút (GridSearchCV)

---

### 6. `stage_05_model_evaluation.py`

**Class**: `ModelEvaluationTrainingPipeline`

**Chức năng**: Orchestrate Stage 5 - Model Evaluation

**Làm gì**:
- Khởi tạo ConfigurationManager
- Lấy config cho model evaluation
- Khởi tạo ModelEvaluation component với config
- Gọi method `initiate_model_evaluation()` để đánh giá model
- Xử lý lỗi và logging

**Có thể chạy độc lập**: `python src/pipeline/stage_05_model_evaluation.py`

---

## 🎯 Mục đích của Pipeline Wrappers

### Tách biệt logic:
- **Component** (trong `src/components/`): Chứa business logic
  - Xử lý dữ liệu, train model, tính metrics, etc.
- **Pipeline** (trong `src/pipeline/`): Chứa orchestration logic
  - Gọi component, xử lý lỗi, logging, quản lý luồng

### Lợi ích:
1. **Modularity**: Mỗi stage độc lập, dễ dàng thêm/xóa/sửa
2. **Testability**: Có thể test từng stage riêng biệt
3. **Debuggability**: Chạy từng stage để xác định lỗi
4. **Reusability**: Component có thể tái sử dụng trong pipeline khác
5. **Clean code**: Tách biệt concerns, dễ maintain

---

## 🔄 Luồng thực thi

### Trong `main.py`:
```
Stage 1: Data Ingestion
    ↓
Stage 2: Data Validation
    ↓
Stage 3: Data Transformation
    ↓
Stage 4: Model Training
    ↓
Stage 5: Model Evaluation
```

Mỗi stage được wrap trong try-except block với logging rõ ràng

---

## 🐛 Debug Tips

**Nếu pipeline bị lỗi**:
1. Chạy từng stage riêng lẻ để xác định stage nào gây lỗi
2. Kiểm tra logs trong thư mục `logs/`
3. Kiểm tra artifacts trong thư mục `artifacts/` để đảm bảo output của stage trước đúng
4. Đọc error message trong console để biết chi tiết lỗi
