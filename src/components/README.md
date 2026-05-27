# 📦 Components - Thư mục xử lý logic

## 📁 Cấu trúc thư mục

```
src/components/
├── data_ingestion.py           # Stage 1: Giải nén dữ liệu
├── data_validation.py          # Stage 2: Kiểm tra schema
├── data_transformation.py      # Stage 3: Feature engineering & preprocessing
├── model_trainer.py            # Stage 4: Training với GridSearchCV
└── model_evaluation.py         # Stage 5: Evaluation và visualization
```

---

## 📄 Mô tả từng file

### 1. `data_ingestion.py` (Stage 1)

**Chức năng chính**: Giải nén dữ liệu từ file zip

**Làm gì**:
- Đọc file `playground-series-s6e3.zip` từ thư mục `data/`
- Giải nén và trích xuất `train.csv` và `test.csv`
- Lưu vào thư mục `artifacts/data_ingestion/`

**Class**: `DataIngestion`  
**Method chính**: `extract_zip_file()`

---

### 2. `data_validation.py` (Stage 2)

**Chức năng chính**: Kiểm tra tính hợp lệ của dữ liệu

**Làm gì**:
- Đọc `train.csv` từ Stage 1
- So sánh các cột trong dữ liệu với schema định nghĩa trong `schema.yaml`
- Kiểm tra xem tất cả các cột có tồn tại và đúng kiểu dữ liệu không
- Ghi kết quả validation (True/False) vào `status.txt`

**Class**: `DataValidation`  
**Method chính**: `validate_all_columns()`

**Output**: `artifacts/data_validation/status.txt`

---

### 3. `data_transformation.py` (Stage 3)

**Chức năng chính**: Tiền xử lý dữ liệu và feature engineering

**Làm gì**:
- **Feature Engineering**: Tạo 9 features mới từ phân tích EDA
  - Ví dụ: `charge_to_tenure_ratio`, `is_high_risk_profile`, `early_churn_flag`, etc.
- **Preprocessing**: 
  - Xử lý missing values (imputation)
  - Xử lý outliers (Winsorization)
  - Chuẩn hóa dữ liệu số (StandardScaler)
  - Encode dữ liệu categorical (OneHotEncoder)
- **SMOTE**: Cân bằng nhãn từ 22.5% Yes → 50% Yes (tăng từ 594k → 920k samples)
- Lưu dữ liệu đã xử lý dạng `.npz` (compressed numpy arrays)
- Lưu preprocessing pipeline dạng `.joblib` để tái sử dụng

**Classes**: 
- `ChurnFeatureEngineer`: Custom transformer tạo features mới
- `WinsorizerTransformer`: Custom transformer xử lý outliers
- `DataTransformation`: Main class orchestrate toàn bộ

**Method chính**: `initiate_data_transformation()`

**Output**: 
- `train_transformed.npz` (920,754 samples, 23 features)
- `test_transformed.npz` (254,655 samples, 23 features)
- `preprocessor.joblib`

---

### 4. `model_trainer.py` (Stage 4)

**Chức năng chính**: Huấn luyện và tối ưu hóa mô hình

**Làm gì**:
- Đọc dữ liệu đã transform từ Stage 3 (file `.npz`)
- Chia tập train thành train (80%) và validation (20%)
- **GridSearchCV** để tìm hyperparameters tốt nhất:
  - LightGBM: 6 tổ hợp tham số
  - XGBoost: 6 tổ hợp tham số
  - Scoring metric: ROC AUC
  - Cross-validation: 3-fold
- So sánh 2 mô hình và chọn mô hình có ROC AUC cao nhất
- Log tất cả experiments vào MLflow (parameters, metrics, models)
- Lưu mô hình tốt nhất

**Class**: `ModelTrainer`  
**Method chính**: `initiate_model_trainer()`

**Output**: 
- `model.joblib` (mô hình tốt nhất)
- `metrics.json` (accuracy, precision, recall, f1, roc_auc)
- MLflow runs

**Thời gian chạy**: ~10-15 phút

---

### 5. `model_evaluation.py` (Stage 5)

**Chức năng chính**: Đánh giá mô hình và tạo visualizations

**Làm gì**:
- Load mô hình đã train từ Stage 4
- Đọc test data từ Stage 3
- Tạo predictions và probabilities trên test set
- **Tính metrics** (nếu test set có nhãn):
  - Accuracy, Precision, Recall, F1-Score, ROC AUC
- **Tạo visualizations**:
  - Confusion Matrix (heatmap)
  - ROC Curve (với AUC score)
- Log tất cả vào MLflow (metrics, artifacts, model registry)

**Class**: `ModelEvaluation`  
**Method chính**: `initiate_model_evaluation()`

**Output**: 
- `predictions.npz` (predictions cho test set)
- `metrics.json` (nếu có nhãn)
- `confusion_matrix.png` (nếu có nhãn)
- `roc_curve.png` (nếu có nhãn)
- MLflow artifacts

**Lưu ý**: Tập test từ Kaggle không có nhãn, nên chỉ tạo predictions

---

## 🔗 Luồng dữ liệu

```
data_ingestion.py
    ↓ train.csv, test.csv
data_validation.py
    ↓ validated data
data_transformation.py
    ↓ train_transformed.npz, test_transformed.npz, preprocessor.joblib
model_trainer.py
    ↓ model.joblib, metrics.json
model_evaluation.py
    ↓ predictions.npz, visualizations
```
