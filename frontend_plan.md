# Kế hoạch Phát triển Frontend – Dự án Customer Churn Prediction

Kế hoạch này vạch ra kiến trúc, giao diện và cách tích hợp 3 tính năng chính vào mẫu giao diện **Horizon UI (React Tailwind CSS)** đã được clone. Giao diện này sẽ kết nối trực tiếp với các API từ FastAPI Backend.

---

## ⚙️ Thiết lập & Kiến trúc chung

- **Framework:** React (phiên bản đi kèm Horizon UI) + Tailwind CSS.
- **Thư viện gọi API:** `axios` để gửi/nhận HTTP requests.
- **Thư viện vẽ biểu đồ:** **ApexCharts** (mặc định đã tích hợp sẵn trong Horizon UI).
- **Cấu hình API Base URL:** Tạo file `.env.local` ở thư mục frontend:
  ```env
  REACT_APP_API_URL=http://127.0.0.1:8000
  ```

### 🗂️ Cấu trúc thư mục định hướng tích hợp
Trong Horizon UI, các trang chính sẽ nằm trong thư mục `src/views/admin/`. Chúng ta sẽ khai báo route trong `src/routes.js` và tạo 3 views tương ứng:
```text
src/
├── routes.js                  # Định nghĩa menu điều hướng sidebar
├── views/
│   └── admin/
│       ├── eda/               # [Tính năng 1] Dashboard phân tích EDA
│       │   └── index.jsx
│       ├── predict/           # [Tính năng 2] Form nhập liệu & Dự đoán Churn
│       │   └── index.jsx
│       └── comparison/        # [Tính năng 3] Huấn luyện & So sánh mô hình
│           └── index.jsx
```

---

## 🗺️ Cấu trúc Menu Điều hướng (Sidebar Navigation)

Cập nhật file `src/routes.js` để hiển thị menu 3 tính năng:
1. **EDA Dashboard:** Phân tích trực quan hóa dữ liệu gốc và các đặc trưng phái sinh.
2. **Churn Prediction:** Nhập hồ sơ khách hàng để dự đoán khả năng rời bỏ dịch vụ.
3. **Model Comparison:** Tự huấn luyện và so sánh hiệu suất các thuật toán ML.

---

## 🛠️ Chi tiết Kế hoạch 3 Tính năng chính

### 📊 Tính năng 1: Giao diện Trực quan hóa dữ liệu (EDA Dashboard) (Bảo)
*Giao diện nhận dữ liệu từ các API `GET /api/v1/eda/...` để hiển thị biểu đồ và nhận xét (insight).*

#### 1. Bố cục trang (Layout Dashboard):
- **Hàng 1: Thẻ KPI nhanh (Mini Statistics Cards):**
  - Số lượng mẫu dữ liệu (Rows).
  - Số lượng thuộc tính (Columns).
  - Số bản ghi trùng lặp (Duplicates).
  - Tổng số ô dữ liệu bị khuyết (Missing values).
- **Hàng 2: Bảng thống kê mô tả (Descriptive Statistics Table):**
  - Hiển thị bảng chứa các chỉ số (`mean`, `min`, `max`, `q1`, `median`, `q3`, `variance`, `skewness`) của các cột số gốc và phái sinh.
- **Hàng 3: Phân phối Đơn biến (Univariate Analysis):**
  - *Bộ lọc:* Thanh Dropdown để chọn cột phân phối (Cột số: `tenure`, `MonthlyCharges`,... Cột phân loại: `Contract`, `InternetService`,...).
  - *Biểu đồ:* 
    - Nếu chọn cột số: Vẽ biểu đồ cột **Histogram** (trục X là khoảng giá trị `labels`, trục Y là `values`) và biểu đồ hộp **Boxplot** từ `boxplot_data`.
    - Nếu chọn cột phân loại: Vẽ biểu đồ tròn **Pie Chart** hoặc **Donut Chart** (dùng `labels` và `percentages`).
- **Hàng 4: Phân tích Đa biến & Tương quan (Bivariate & Correlation):**
  - *Bên trái:* Đồ thị nhiệt **Correlation Heatmap** (`GET /api/v1/eda/correlation`) thể hiện sự tương quan giữa các đặc trưng số.
  - *Bên phải:* Biểu đồ phân tích chéo tương tác với `Churn` (`GET /api/v1/eda/bivariate/{feature_name}`). Hỗ trợ vẽ **Stacked Bar Chart** (nếu là cột chữ) hoặc **Side-by-Side Boxplot** (nếu là cột số).
- **Phần cuối:** Khung hiển thị chuỗi nhận xét `"insight"` lấy động từ API tương ứng.

---

### 🔮 Tính năng 2: Giao diện Dự đoán Churn đơn lẻ (Churn Prediction) (Phát)
*Nhận thông tin 1 khách hàng từ form nhập liệu, gửi request POST lên `/api/v1/predict` (hoặc endpoint tương tự của Phát) và hiển thị kết quả dự đoán kèm xác suất.*

#### 1. Thiết kế Form Nhập liệu (Customer Profile Input Form):
Chia form thành 3 nhóm trường rõ ràng (sử dụng Layout Grid 3 cột của Tailwind):
- **Nhóm 1: Nhân khẩu học (Demographics):**
  - `gender` (Dropdown: Male / Female)
  - `SeniorCitizen` (Dropdown: Yes / No)
  - `Partner` (Dropdown: Yes / No)
  - `Dependents` (Dropdown: Yes / No)
- **Nhóm 2: Dịch vụ đăng ký (Services):**
  - `PhoneService` & `MultipleLines` (Dropdowns)
  - `InternetService` (Dropdown: DSL / Fiber optic / No)
  - Các dịch vụ giá trị gia tăng (`OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`) -> Thiết kế ở dạng Dropdown/Checkbox (Chỉ kích hoạt chọn nếu `InternetService` khác `No`).
- **Nhóm 3: Tài chính & Hợp đồng (Financials):**
  - `Contract` (Dropdown: Month-to-month / One year / Two year)
  - `PaperlessBilling` (Dropdown: Yes / No)
  - `PaymentMethod` (Dropdown: Electronic check / Mailed check / Bank transfer (automatic) / Credit card (automatic))
  - `tenure` (Input số nguyên từ 0 đến 100)
  - `MonthlyCharges` & `TotalCharges` (Input số thực)

#### 2. Hiển thị Kết quả Dự đoán (Prediction Result Card):
Hiển thị ngay bên cạnh hoặc phía dưới form sau khi người dùng bấm nút **"Predict Churn"**:
- **Trạng thái rủi ro:** Hiển thị thẻ màu động:
  - **Màu đỏ + Icon Cảnh báo:** Nếu kết quả là `Yes` (Khách hàng có xu hướng rời bỏ dịch vụ).
  - **Màu xanh + Icon An toàn:** Nếu kết quả là `No` (Khách hàng tiếp tục gắn bó).
- **Thước đo Xác suất Churn (Probability Gauge):**
  - Sử dụng biểu đồ Radial Bar hoặc thanh Progress bar của Tailwind chạy từ 0% đến 100% để thể hiện giá trị xác suất dự đoán (ví dụ: `78.5%` rủi ro rời mạng).

---

### ⚖️ Tính năng 3: Giao diện So sánh & Huấn luyện Mô hình (Model Comparison) (Lan Anh)
*Giao diện tương tác gửi các tham số huấn luyện lên API `/api/v1/compare-models` (hoặc endpoint tương ứng của Khang) để tiến hành huấn luyện nhanh và đối chiếu các thuật toán.*

#### 1. Bộ cấu hình tham số Huấn luyện (Training Panel):
- **Chọn thuật toán (Model Selection):** Dropdown gồm các mô hình được hỗ trợ (`Logistic Regression`, `Random Forest`, `XGBoost`, `LightGBM`).
- **Tùy chỉnh siêu tham số (Hyperparameters):** Các ô input động thay đổi theo mô hình:
  - Cần chỉnh `max_depth` (độ sâu cây), `n_estimators` (số cây quyết định), `learning_rate` (tốc độ học), `C` (hệ số điều chuẩn).
- **Thông số phân tập dữ liệu:**
  - Tỷ lệ Validation/Test size (Slider chạy từ `0.1` đến `0.4`, mặc định `0.2`).
  - Quy mô tập mẫu rút gọn - `sample_size` (Slider từ `1000` đến `50000`).
- Nút bấm **"Train Model"** hiển thị hiệu ứng Loading/Spinner khi API đang xử lý.

#### 2. Dashboard kết quả đánh giá (Model Performance Board):
- **Hàng 1: Thẻ chỉ số chất lượng (Metrics Cards):**
  - Accuracy (Độ chính xác toàn cục)
  - Precision (Độ chính xác dự báo Churn)
  - Recall (Khả năng bắt trúng khách hàng Churn)
  - F1-Score (Chỉ số hài hòa F1)
  - ROC-AUC (Khả năng phân hóa của mô hình)
- **Hàng 2: Ma trận nhầm lẫn (Confusion Matrix Diagram):**
  - Biểu diễn ma trận nhầm lẫn 2 chiều `[[TN, FP], [FN, TP]]` dưới dạng bảng hoặc biểu đồ nhiệt 2x2.
- **Hàng 3: Bảng lịch sử huấn luyện (Comparison Leaderboard Table):**
  - Lưu lại kết quả của các lượt huấn luyện trước đó để người dùng so sánh trực quan xem thuật toán nào, với tham số nào thì đạt điểm tối ưu nhất.

---

## 🏃 Quy trình cài đặt & Vận hành cục bộ

1. **Khởi chạy Frontend:**
   Di chuyển vào thư mục dự án của Horizon UI và khởi động server phát triển:
   ```bash
   npm install
   npm run start
   ```
2. **Khởi chạy Backend:**
   Đảm bảo server FastAPI đang chạy ở cổng 8000:
   ```bash
   cd backend
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```
3. Mở trình duyệt truy cập `http://localhost:3000` (hoặc cổng do Horizon UI cấp phát) để kiểm tra giao diện tích hợp.
