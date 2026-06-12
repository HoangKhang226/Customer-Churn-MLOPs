# Kế hoạch Phát triển Backend – Dự án Customer Churn Prediction Web App

## Tổng quan

- **Kiến trúc:** Tách biệt Frontend (HTML/JS Framework) và Backend (FastAPI)
- **Số lượng thành viên Backend:** 3 người
- **Framework Backend:** FastAPI + Uvicorn
- **Ngôn ngữ:** Python 3.10+
- **Phân công:** Mỗi người phụ trách 1 tính năng chính

| Người | Tính năng phụ trách |
|-------|---------------------|
| **Phát** | Dự đoán Churn đơn lẻ (Prediction API) |
| **Đức** | Trực quan hóa EDA (EDA Data API) |
| **Khang** | Sử dụng các mô hình khác nhau để đưa ra các chỉ số đánh giá, so sánh các mô hình  |

---

## Cấu trúc thư mục Backend

```text
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                  # Khởi chạy FastAPI, cấu hình CORS, gắn các router
│   ├── config.py                # Đường dẫn tới model, preprocessor, dữ liệu
│   │
│   ├── schemas/                 # Pydantic schemas – Xác thực dữ liệu đầu vào/đầu ra
│   │   ├── __init__.py
│   │   ├── predict_schema.py    # [Người 1] Schema cho request/response dự đoán
│   │   ├── eda_schema.py        # [Người 2] Schema cho response thống kê EDA
│   │   └── train_schema.py      # [Người 3] Schema cho request/response huấn luyện
│   │
│   ├── routes/                  # API Endpoints (Router)
│   │   ├── __init__.py
│   │   ├── predict_route.py     # [Người 1] POST /api/predict
│   │   ├── eda_route.py         # [Người 2] GET  /api/eda/...
│   │   └── compare_model_route.py       # [Người 3] POST /api/compare-models
│   │
│   ├── services/                # Logic nghiệp vụ chính
│   │   ├── __init__.py
│   │   ├── predict_service.py   # [Người 1] Load model, tiền xử lý, suy luận
│   │   ├── eda_service.py       # [Người 2] Tính toán thống kê, tương quan, phân phối
│   │   └── compare_model_service.py     # [Người 3] Lấy mẫu, huấn luyện nhanh, tính metrics
│   │
│   └── ml_artifacts/            # Chứa file model và preprocessor đã huấn luyện offline
│       ├── model.joblib          # Mô hình tốt nhất từ pipeline offline
│       └── preprocessor.joblib   # Pipeline tiền xử lý (FeatureEngineer + ColumnTransformer)
│
├── data/                        # Dữ liệu gốc cho EDA và Training
│   └── train.csv
│
├── tests/                       # Unit test cho từng tính năng
│   ├── __init__.py
│   ├── test_predict.py          # [Người 1]
│   ├── test_eda.py              # [Người 2]
│   └── test_train.py            # [Người 3]
│
├── requirements.txt
├── Dockerfile
└── README.md
```

### Quy ước chung cho cả 3 người

- **File `app/main.py`** là file dùng chung, do bất kỳ ai tạo trước. File này chỉ chứa logic khởi tạo FastAPI và `include_router` cho các route. Khi mỗi người hoàn thành route của mình, chỉ cần thêm 1 dòng `app.include_router(...)` vào file này.
- **File `app/config.py`** là file dùng chung, chứa các hằng số đường dẫn (path tới model, preprocessor, dữ liệu).
- Mỗi người chỉ chỉnh sửa các file trong phạm vi phụ trách của mình (đánh dấu `[Người X]` ở cấu trúc thư mục trên).

---

## Người 1 – API Dự đoán Churn (Prediction)

### Mô tả
Xây dựng endpoint nhận thông tin 1 khách hàng, đưa qua pipeline tiền xử lý và mô hình đã lưu sẵn (`model.joblib`, `preprocessor.joblib`) để trả về kết quả dự đoán Churn (Yes/No) cùng xác suất rời bỏ.

### Công cụ sử dụng
- **FastAPI** – Viết API endpoint
- **Pydantic** – Xác thực dữ liệu đầu vào từ người dùng
- **Joblib** – Load file model và preprocessor
- **Pandas** – Chuyển đổi dữ liệu đầu vào thành DataFrame
- **Swagger UI** (tự động từ FastAPI) – Kiểm tra API trên trình duyệt
- **Pytest + httpx** – Viết unit test

### Các bước thực hiện

1. **Tạo Pydantic schema** (`schemas/predict_schema.py`):
   - Khai báo class `CustomerInput` chứa tất cả các trường thông tin khách hàng với kiểu dữ liệu và giá trị mặc định phù hợp (ví dụ: `tenure: int`, `MonthlyCharges: float`, `Contract: str`).
   - Khai báo class `PredictionResponse` chứa kết quả trả về: `churn_prediction: str` (Yes/No), `churn_probability: float` (0.0 đến 1.0).

2. **Viết service xử lý logic** (`services/predict_service.py`):
   - Viết hàm `load_model()` để nạp `model.joblib` và `preprocessor.joblib` từ thư mục `ml_artifacts/`. Sử dụng cơ chế cache (biến global hoặc `functools.lru_cache`) để chỉ load 1 lần khi server khởi động.
   - Viết hàm `predict_churn(customer_data: dict) -> dict`:
     - Chuyển `customer_data` thành DataFrame 1 dòng.
     - Loại bỏ các cột đã bị drop trong pipeline gốc (`id`, `TotalCharges`, `gender`).
     - Gọi `preprocessor.transform()` để chuẩn hóa dữ liệu.
     - Gọi `model.predict_proba()` để lấy xác suất.
     - Trả về dictionary chứa nhãn dự đoán và xác suất.

3. **Viết route** (`routes/predict_route.py`):
   - Tạo `APIRouter` với prefix `/api`.
   - Khai báo endpoint `POST /api/predict` nhận body dạng `CustomerInput`, gọi service và trả về `PredictionResponse`.

4. **Viết unit test** (`tests/test_predict.py`):
   - Test case 1: Gửi dữ liệu hợp lệ đầy đủ → Kỳ vọng mã 200, kết quả có trường `churn_prediction` và `churn_probability`.
   - Test case 2: Gửi dữ liệu thiếu trường bắt buộc → Kỳ vọng mã 422 (Validation Error).
   - Test case 3: Gửi giá trị không hợp lệ (ví dụ `tenure` là chuỗi chữ) → Kỳ vọng mã 422.
   - Test case 4: Xác nhận `churn_probability` nằm trong khoảng [0.0, 1.0].

### Sản phẩm đầu ra
| File | Mô tả |
|------|-------|
| `schemas/predict_schema.py` | Pydantic model xác thực input/output |
| `services/predict_service.py` | Logic load model và dự đoán |
| `routes/predict_route.py` | Endpoint `POST /api/predict` |
| `tests/test_predict.py` | Bộ test tự động (≥ 4 test cases) |

### Tiêu chí nghiệm thu
- [ ] Endpoint `POST /api/predict` trả về mã 200 khi nhận dữ liệu hợp lệ.
- [ ] Kết quả trả về đúng cấu trúc JSON: `{ "churn_prediction": "Yes"|"No", "churn_probability": 0.xx }`.
- [ ] Xác suất `churn_probability` nằm trong khoảng `[0.0, 1.0]`.
- [ ] API từ chối dữ liệu thiếu trường hoặc sai kiểu với mã lỗi 422 và thông báo lỗi rõ ràng.
- [ ] Thời gian phản hồi dưới **500ms** cho mỗi request dự đoán.
- [ ] Tất cả test cases trong `test_predict.py` chạy pass 100%.

---

## Người 2 – API Dữ liệu EDA (EDA Dashboard)

### Mô tả
Xây dựng các endpoint trả về dữ liệu thống kê dưới dạng JSON và đoạn phân tích định tính (insight) thu gọn để phía Frontend nhận, vẽ biểu đồ tương tác và hiển thị nhận xét. Backend **không vẽ biểu đồ**, chỉ tính toán và cung cấp dữ liệu số liệu kèm insight.

### Công cụ sử dụng
- **FastAPI** – Viết API endpoint
- **Pandas** – Tính toán thống kê mô tả, tương quan, phân phối
- **NumPy** – Hỗ trợ tính toán ma trận tương quan và phân phối histogram
- **Swagger UI** – Kiểm tra API
- **Pytest + httpx** – Viết unit test

### Các bước thực hiện

1. **Tạo Pydantic schema** (`schemas/eda_schema.py`):
   - Định nghĩa cấu trúc response có chứa trường `"insight": str` cho tất cả các endpoint:
     - `DatasetOverviewResponse`: Tổng quan dữ liệu (rows, columns, duplicates, missing values, column_types, feature_roles).
     - `SanityCheckResponse`: Kiểm tra tính hợp lệ logic nghiệp vụ (numerical_sanity, categorical_sanity).
     - `NumericalStatsResponse`: Các chỉ số thống kê đơn biến của toàn bộ các cột số (tenure, MonthlyCharges, TotalCharges).
     - `NumericalDistributionResponse`: Nhãn khoảng cước và tần suất cột số để Frontend vẽ Histogram cùng 5 chỉ số vẽ Boxplot.
     - `CategoricalDistributionResponse`: Phân phối tần suất (%) của cột phân loại (bao gồm cả target Churn).
     - `BivariateAnalysisResponse`: Phân tích chéo tần suất (cột phân loại) hoặc so sánh chỉ số phân nhóm (cột số) liên kết với Churn.
     - `CorrelationMatrixResponse`: Ma trận hệ số tương quan Pearson.

2. **Viết service xử lý logic** (`services/eda_service.py`):
   - Đọc file `train.csv` và chuyển đổi cột SeniorCitizen thành 'Yes'/'No'.
   - Viết các hàm tính toán kèm log chi tiết bằng thư viện `logging`:
     - `get_dataset_overview()`
     - `get_data_sanity_check()`
     - `get_numerical_statistics()`
     - `get_numerical_distribution(column_name, bins)`
     - `get_categorical_distribution(column_name)`
     - `get_bivariate_analysis(feature_name)`
     - `get_correlation_matrix()`

3. **Viết route** (`routes/eda_route.py`):
   - Đăng ký các endpoints với router `/api/v1/eda`:
     - `GET /api/v1/eda/overview`
     - `GET /api/v1/eda/sanity-check`
     - `GET /api/v1/eda/numerical-stats`
     - `GET /api/v1/eda/distribution/numerical/{column_name}`
     - `GET /api/v1/eda/distribution/categorical/{column_name}`
     - `GET /api/v1/eda/bivariate/{feature_name}`
     - `GET /api/v1/eda/correlation`

4. **Viết unit test** (`tests/test_eda.py`):
   - Test case 1: Gọi `GET /api/v1/eda/overview` → Mã 200, response chứa `shape` và `insight`.
   - Test case 2: Gọi `GET /api/v1/eda/sanity-check` → Mã 200, chứa lỗi logic nghiệp vụ.
   - Test case 3: Gọi `GET /api/v1/eda/numerical-stats` → Mã 200, chứa thống kê 3 cột số.
   - Test case 4: Gọi `GET /api/v1/eda/distribution/numerical/MonthlyCharges` → Mã 200, chứa nhãn bins và boxplot.
   - Test case 5: Gọi `GET /api/v1/eda/bivariate/PaymentMethod` → Mã 200, chứa bảng phân tích chéo tần suất.
   - Test case 6: Gọi `GET /api/v1/eda/correlation` → Mã 200, chứa ma trận vuông N×N.
   - Test case 7: Gọi sai tên cột `GET /api/v1/eda/bivariate/INVALID_COL` → Trả lỗi 400.

### Sản phẩm đầu ra
| File | Mô tả |
|------|-------|
| `schemas/eda_schema.py` | Pydantic models cho response API có insight |
| `services/eda_service.py` | Logic tính toán số liệu và ghi log |
| `routes/eda_route.py` | 7 endpoints GET `/api/v1/eda/...` |
| `tests/test_eda.py` | Bộ test tự động (≥ 7 test cases) |

### Tiêu chí nghiệm thu
- [ ] Tất cả 7 endpoint trả về mã 200 khi truyền dữ liệu hợp lệ.
- [ ] Dữ liệu JSON trả về có cấu trúc rõ ràng, kèm trường `"insight"` giải thích chi tiết ý nghĩa từ notebook.
- [ ] Hệ thống trả mã lỗi 400 Bad Request kèm thông báo chi tiết khi truyền cột không tồn tại.
- [ ] Thời gian phản hồi trung bình các endpoint dưới **1.5 giây**.
- [ ] Tất cả test cases trong `test_eda.py` chạy pass 100%.

---

## Người 3 – So sánh các mô hình (Model Comparison)

### Mô tả
Xây dựng endpoint cho phép người dùng chọn thuật toán phân loại, thiết lập siêu tham số, rồi tiến hành huấn luyện nhanh trên một tập mẫu rút gọn (downsample) của dữ liệu gốc. Trả về các chỉ số đánh giá và ma trận nhầm lẫn.

### Công cụ sử dụng
- **FastAPI** – Viết API endpoint
- **Pydantic** – Xác thực tham số huấn luyện từ người dùng
- **Scikit-learn** – Logistic Regression, Decision Tree, Random Forest
- **XGBoost / LightGBM** – Thuật toán Gradient Boosting
- **Pandas + NumPy** – Xử lý dữ liệu
- **Swagger UI** – Kiểm tra API
- **Pytest + httpx** – Viết unit test

### Các bước thực hiện

1. **Tạo Pydantic schema** (`schemas/train_schema.py`):
   - Khai báo class `TrainRequest`:
     - `model_type: str` – Thuật toán (ví dụ: `"logistic_regression"`, `"random_forest"`, `"xgboost"`, `"lightgbm"`).
     - `hyperparameters: dict` – Siêu tham số tùy chỉnh (ví dụ: `{ "n_estimators": 100, "max_depth": 5 }`).
     - `test_size: float` – Tỷ lệ chia tập validation (mặc định `0.2`).
     - `sample_size: int` – Số dòng lấy mẫu để huấn luyện nhanh (mặc định `30000`, tối đa `50000`).
   - Khai báo class `TrainResponse`:
     - `model_type: str`, `accuracy: float`, `precision: float`, `recall: float`, `f1_score: float`, `roc_auc: float`.
     - `confusion_matrix: list[list[int]]` – Ma trận nhầm lẫn dạng mảng 2 chiều [[TN, FP], [FN, TP]].
     - `training_time_seconds: float` – Thời gian huấn luyện (giây).

2. **Viết service xử lý logic** (`services/train_service.py`):
   - Viết hàm `load_and_sample_data(sample_size)`:
     - Đọc file `train.csv`.
     - Lấy mẫu ngẫu nhiên `sample_size` dòng (stratified theo biến Churn để giữ tỷ lệ mất cân bằng).
     - Thực hiện tiền xử lý giống pipeline gốc (load `preprocessor.joblib` và transform).
     - Chia Train/Validation theo `test_size`.
   - Viết hàm `get_model(model_type, hyperparameters)`:
     - Trả về đối tượng model tương ứng với thuật toán được chọn, đã áp dụng hyperparameters.
     - Nếu `model_type` không nằm trong danh sách hỗ trợ → raise lỗi 400.
   - Viết hàm `train_and_evaluate(request: TrainRequest) -> dict`:
     - Gọi `load_and_sample_data()`, `get_model()`.
     - Đo thời gian huấn luyện bằng `time.time()`.
     - Tính Accuracy, Precision, Recall, F1, ROC-AUC.
     - Tính Confusion Matrix.
     - Trả về dictionary chứa tất cả thông tin trên.

3. **Viết route** (`routes/compare_model_route.py`):
   - `POST /api/compare-models` nhận body dạng `TrainRequest`, gọi service và trả về `TrainResponse`.
   - `GET /api/compare-models/models` trả về danh sách các thuật toán được hỗ trợ cùng các siêu tham số mặc định.

4. **Viết unit test** (`tests/test_train.py`):
   - Test case 1: Gửi request hợp lệ với `model_type = "logistic_regression"` → Kỳ vọng mã 200, đầy đủ metrics.
   - Test case 2: Gửi request hợp lệ với `model_type = "xgboost"` → Kỳ vọng mã 200.
   - Test case 3: Gửi `model_type = "invalid_model"` → Kỳ vọng mã 400.
   - Test case 4: Gửi `sample_size = 100000` (vượt giới hạn 50000) → Kỳ vọng mã 422.
   - Test case 5: Xác nhận `training_time_seconds < 30` (mô hình phải train xong dưới 30 giây).

### Sản phẩm đầu ra
| File | Mô tả |
|------|-------|
| `schemas/train_schema.py` | Pydantic models cho request/response training |
| `services/train_service.py` | Logic lấy mẫu, khởi tạo model, train, đánh giá |
| `routes/train_route.py` | Endpoint `POST /api/train` và `GET /api/train/models` |
| `tests/test_train.py` | Bộ test tự động (≥ 5 test cases) |

### Tiêu chí nghiệm thu
- [ ] Endpoint `POST /api/train` trả về mã 200 cho tất cả thuật toán hỗ trợ (Logistic Regression, Random Forest, XGBoost, LightGBM).
- [ ] Kết quả trả về đúng cấu trúc JSON với đầy đủ 6 chỉ số đánh giá và confusion matrix.
- [ ] Tất cả metrics (accuracy, precision, recall, f1, roc_auc) nằm trong khoảng `[0.0, 1.0]`.
- [ ] Thời gian huấn luyện dưới **30 giây** với `sample_size = 30000`.
- [ ] API trả lỗi 400 khi `model_type` không được hỗ trợ.
- [ ] API trả lỗi 422 khi `sample_size` vượt quá giới hạn cho phép.
- [ ] Tất cả test cases trong `test_train.py` chạy pass 100%.

---

## Phần dùng chung (Bất kỳ ai trong nhóm tạo trước)

### File `app/main.py`
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import predict_route, eda_route, train_route

app = FastAPI(title="Customer Churn Prediction API", version="1.0.0")

# Cho phép Frontend truy cập API từ domain khác
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gắn router của từng tính năng
app.include_router(predict_route.router)
app.include_router(eda_route.router)
app.include_router(train_route.router)

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Churn Prediction API is running"}
```

### File `app/config.py`
```python
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ML_ARTIFACTS_DIR = BASE_DIR / "ml_artifacts"
DATA_DIR = BASE_DIR.parent / "data"

MODEL_PATH = ML_ARTIFACTS_DIR / "model.joblib"
PREPROCESSOR_PATH = ML_ARTIFACTS_DIR / "preprocessor.joblib"
TRAIN_DATA_PATH = DATA_DIR / "train.csv"
```

### File `requirements.txt`
```text
fastapi>=0.100.0
uvicorn>=0.23.0
pydantic>=2.0.0
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.2.0
xgboost>=1.7.0
lightgbm>=3.3.0
joblib>=1.2.0
httpx>=0.24.0
pytest>=7.0.0
```

### Lệnh chạy server cục bộ
```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Server sẽ chạy tại `http://127.0.0.1:8000`.
Tài liệu Swagger tự động tại `http://127.0.0.1:8000/docs`.

### Lệnh chạy toàn bộ test
```bash
cd backend
pytest tests/ -v
```
