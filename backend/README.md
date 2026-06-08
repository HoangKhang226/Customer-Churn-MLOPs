# 🚀 FastAPI Backend – Hướng dẫn Vận hành & Tài liệu tích hợp API (EDA)

Thư mục này chứa mã nguồn của Backend API dự án **Customer Churn Prediction** được xây dựng bằng **FastAPI** và chạy trên máy chủ **Uvicorn**.

---

## 1. Hướng dẫn khởi chạy & Xem thử API trực quan (Swagger UI)

### 💻 Khởi chạy Server Backend
Đảm bảo bạn đã cài đặt các thư viện cần thiết trong file [requirements.txt](file:///c:/Users/Admin/Documents/viet_code/python/NMKHDL/customer_churn_prediction/requirements.txt):
```bash
pip install -r requirements.txt
```

Khởi chạy máy chủ phát triển cục bộ từ thư mục gốc của dự án:
```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
*   **Địa chỉ Server:** `http://127.0.0.1:8000`
*   **Kiểm tra hoạt động:** Truy cập `http://127.0.0.1:8000/` (Sẽ trả về chuỗi JSON thông báo server đang hoạt động).

### 🔍 Cách kiểm tra và xem thử API trực quan
FastAPI tự động sinh tài liệu API tương tác bằng **Swagger UI** và **ReDoc**. Để chạy thử trực tiếp:
1. Mở trình duyệt và truy cập: **`http://127.0.0.1:8000/docs`**
2. Tại đây, bạn sẽ thấy toàn bộ danh sách các API định tuyến dưới tag **Exploratory Data Analysis**.
3. **Cách xem thử (Try it out):**
   * Click vào một API bất kỳ để mở rộng chi tiết.
   * Click vào nút **Try it out** ở góc phải.
   * Nhập tham số nếu có (ví dụ: `column_name` hoặc `feature_name`).
   * Click vào nút **Execute** màu xanh da trời.
4. **Xem kết quả:** Kết quả trả về thực tế (JSON), mã trạng thái (HTTP Status Code 200, 422, 500,...) và Headers sẽ hiển thị trực quan ngay tại mục **Response body**.

---

## 2. Tài liệu chi tiết các trường dữ liệu Response cho Frontend (HTML/JS)

Mọi API phân tích khám phá dữ liệu (EDA) đều có trường `"insight"` chứa nội dung nhận xét/phân tích chuyên sâu từ dữ liệu dưới dạng chuỗi (String). Frontend có thể hiển thị trực tiếp chuỗi này lên giao diện làm chú thích hoặc giải thích cho biểu đồ.

Dưới đây là chi tiết các trường dữ liệu và gợi ý cách vẽ đồ thị tương ứng cho từng API:

### 📊 1. Tổng quan tập dữ liệu
*   **Endpoint:** `GET /api/v1/eda/overview`
*   **Ý nghĩa các trường trong Response:**
    *   `shape` (Object): Gồm `rows` (Số lượng dòng) và `columns` (Số lượng cột).
    *   `duplicates` (Int): Số lượng bản ghi bị trùng lặp dữ liệu hoàn toàn.
    *   `missing_values_count` (Int): Tổng số ô dữ liệu bị thiếu/trống (`NaN`, `None`, `null`).
    *   `column_types` (Object): Map dạng `{ "tên_cột": "kiểu_dữ_liệu" }` để biết kiểu của từng cột.
    *   `feature_roles` (Object): Phân loại vai trò của các cột:
        *   `identifiers`: Danh sách cột định danh (như `customerID`).
        *   `numerical`: Danh sách các cột số (như `tenure`, `MonthlyCharges`, `TotalCharges`).
        *   `categorical`: Danh sách các cột phân loại/chữ (như `gender`, `Contract`, `InternetService`,...).
        *   `target`: Danh sách cột mục tiêu cần dự đoán (ở đây là `['Churn']`).
*   **💡 Lưu ý Frontend:** 
    *   Hiển thị thông số `shape`, `duplicates`, `missing_values_count` lên thẻ thống kê nhanh (KPI Cards).
    *   Sử dụng danh sách `feature_roles.numerical` và `feature_roles.categorical` để tự động điền (populate) vào thanh dropdown lựa chọn của người dùng khi họ muốn chọn cột để xem biểu đồ phân phối/tương quan.

---

### 🔍 2. Kiểm tra logic dữ liệu (Sanity Check)
*   **Endpoint:** `GET /api/v1/eda/sanity-check`
*   **Ý nghĩa các trường trong Response:**
    *   `numerical_sanity` (Object): Chứa số lỗi logic dữ liệu số:
        *   `tenure_invalid` (Int): Số dòng có số tháng sử dụng `<= 0` hoặc là số lẻ không nguyên.
        *   `monthly_charges_invalid` (Int): Số dòng lỗi cước phí tháng `<= 0`.
        *   `total_charges_invalid` (Int): Số dòng lỗi tổng cước `<= 0`.
    *   `categorical_sanity` (Object): Chứa lỗi logic dữ liệu phân loại:
        *   `internet_logic_errors` (Int): Số dòng ghi nhận có dịch vụ đi kèm như `OnlineSecurity` nhưng `InternetService` lại là `No` (lỗi mâu thuẫn dịch vụ).
*   **💡 Lưu ý Frontend:** Hiển thị các chỉ số này dưới dạng danh sách cảnh báo hoặc bảng kiểm tra chất lượng dữ liệu (Data Quality Audit). Nếu tất cả các chỉ số đều bằng `0`, thông báo tập dữ liệu hoàn toàn sạch và logic.

---

### 🧮 3. Thống kê mô tả các cột số (Descriptive Statistics)
*   **Endpoint:** `GET /api/v1/eda/numerical-stats`
*   **Ý nghĩa các trường trong Response:** Trả về thống kê của 3 cột số quan trọng (`tenure`, `MonthlyCharges`, `TotalCharges`). Mỗi cột là một Object chứa các trường:
    *   `mean`: Giá trị trung bình của cột.
    *   `min` / `max`: Giá trị nhỏ nhất / lớn nhất.
    *   `q1` / `median` / `q3`: Phân vị 25% / Trung vị (50%) / Phân vị 75%.
    *   `variance`: Phương sai dữ liệu.
    *   `skewness`: Độ lệch phân phối (nếu `> 0` lệch phải, `< 0` lệch trái, `= 0` đối xứng).
    *   `nunique`: Số lượng giá trị duy nhất (độc nhất).
*   **💡 Lưu ý Frontend:** Thiết kế một bảng dữ liệu (Data Table) với các hàng là tên cột (`tenure`, `MonthlyCharges`, `TotalCharges`) và các cột là các chỉ số thống kê trên để hiển thị so sánh.

---

### 📈 4. Phân phối tần suất cột số (Histogram & Boxplot)
*   **Endpoint:** `GET /api/v1/eda/distribution/numerical/{column_name}`
*   **Ý nghĩa các trường trong Response:**
    *   `labels` (List[Str]): Nhãn hiển thị cho các khoảng chia giá trị (ví dụ: `["18.8-28.8", "28.8-38.8", ...]`).
    *   `values` (List[Int]): Số lượng khách hàng có giá trị rơi vào khoảng tương ứng.
    *   `boxplot_data` (Object): Chứa 5 chỉ số vẽ đồ thị hộp:
        *   `min` / `max`: Giá trị biên dưới / biên trên của hộp.
        *   `q1` / `median` / `q3`: Các phân vị tương ứng của hộp.
*   **💡 Lưu ý Frontend:**
    *   Vẽ **Histogram (Bar Chart)**: Trục X truyền `labels`, trục Y truyền `values`.
    *   Vẽ **Boxplot (Biểu đồ hộp)**: Sử dụng các giá trị trong `boxplot_data` để dựng biểu đồ hộp nhằm chỉ ra phạm vi biến thiên và phát hiện các điểm ngoại lai (outliers).

---

### 🥧 5. Phân phối tần suất cột chữ/phân loại
*   **Endpoint:** `GET /api/v1/eda/distribution/categorical/{column_name}`
*   **Ý nghĩa các trường trong Response:**
    *   `labels` (List[Str]): Danh sách các giá trị phân loại của cột (ví dụ với `gender`: `["Male", "Female"]`).
    *   `counts` (List[Int]): Số lượng khách hàng thuộc từng nhóm.
    *   `percentages` (List[Float]): Tỷ lệ phần trăm tương ứng của từng nhóm (đơn vị %).
*   **💡 Lưu ý Frontend:**
    *   Vẽ biểu đồ tròn **Pie Chart** hoặc **Donut Chart** để biểu diễn cơ cấu thành phần (dùng `labels` và `percentages`).
    *   Hoặc vẽ biểu đồ cột **Bar Chart** nằm ngang để thể hiện quy mô (dùng `labels` và `counts`).

---

### 🔗 6. Phân tích đa biến với thuộc tính mục tiêu Churn (Bivariate Analysis)
*   **Endpoint:** `GET /api/v1/eda/bivariate/{feature_name}`
*   **Lưu ý cực kỳ quan trọng cho Frontend:** API này trả về cấu trúc động tùy thuộc vào kiểu dữ liệu của `{feature_name}`. Frontend cần kiểm tra trường `"type"` trong JSON nhận được để hiển thị giao diện phù hợp:
    
    #### THƯỜNG HỢP 1: Biến phân loại (`type == "categorical"`)
    *   `index` (List[Str]): Danh sách các giá trị phân loại của biến (ví dụ: `["Month-to-month", "One year", "Two year"]`).
    *   `columns` (List[Str]): Nhãn cột mục tiêu Churn (luôn là `["No", "Yes"]`).
    *   `values` (List[List[Int]]): Mảng 2 chiều chứa bảng chéo tần suất. Ví dụ: `[[100, 200], [400, 50], ...]` tương ứng với số lượng Churn (No/Yes) của từng nhóm trong `index`.
    *   **📊 Vẽ biểu đồ:** Vẽ biểu đồ cột chồng **Stacked Bar Chart** hoặc cột nhóm **Grouped Bar Chart** để so sánh tỷ lệ rời bỏ dịch vụ giữa các nhóm khách hàng khác nhau.
    
    #### THƯỜNG HỢP 2: Biến số (`type == "numerical"`)
    *   `churn_yes_stats` (Object) & `churn_no_stats` (Object): Thống kê mô tả riêng cho nhóm rời bỏ (Yes) và ở lại (No). Mỗi đối tượng gồm:
        *   `mean` (Float): Giá trị trung bình của biến đối với nhóm đó.
        *   `boxplot` (List[Float]): Mảng 5 số `[min, q1, median, q3, max]` để vẽ Boxplot.
    *   **📊 Vẽ biểu đồ:** Vẽ biểu đồ hộp đôi nằm cạnh nhau **Side-by-Side Boxplot** hoặc vẽ 2 biểu đồ phân phối Histogram xếp đè (Overlaid Histograms) để so sánh hành vi tiêu dùng giữa khách hàng rời đi và khách hàng ở lại.

---

### 🌡️ 7. Ma trận tương quan (Correlation Matrix)
*   **Endpoint:** `GET /api/v1/eda/correlation`
*   **Ý nghĩa các trường trong Response:**
    *   `columns` (List[Str]): Danh sách các cột số trong ma trận (`["tenure", "MonthlyCharges", "TotalCharges"]`).
    *   `index` (List[Str]): Danh sách nhãn hàng (tương tự như `columns`).
    *   `values` (List[List[Float]]): Mảng 2 chiều kích thước `NxN` chứa hệ số tương quan tuyến tính (Pearson r) có giá trị từ `-1.0` đến `1.0`.
*   **💡 Lưu ý Frontend:** Vẽ biểu đồ nhiệt **Heatmap (Correlation Matrix)**. Sử dụng dải màu tương phản (ví dụ: màu đỏ cho tương quan dương mạnh `+1.0`, màu xanh cho tương quan âm mạnh `-1.0`, và màu trắng/trung tính cho không tương quan `0.0`). Hiển thị trị số hệ số tương quan lên từng ô.

---

## 3. Hướng dẫn & Quy định kỹ thuật dành cho nhóm Backend

Để đảm bảo dự án chạy ổn định, đồng bộ và dễ tích hợp, các thành viên phát triển Backend cần tuân thủ nghiêm ngặt các quy tắc thiết kế sau:

### 📐 1. Kiến trúc phân tách 3 lớp (Layered Architecture)
Tất cả các tính năng mới phải được tổ chức thành 3 lớp rõ rệt:
*   **Schemas (`backend/app/schemas/`)**:
    *   Định nghĩa dữ liệu đầu vào (Request) và đầu ra (Response) bằng `pydantic.BaseModel`.
    *   **Bắt buộc** ghi rõ mô tả cho từng trường bằng tham số `Field(..., description="Mô tả tiếng Việt")`. Điều này giúp Swagger UI tự động hiển thị tài liệu tiếng Việt đầy đủ cho Frontend mà không cần viết file tài liệu bên ngoài.
*   **Services (`backend/app/services/`)**:
    *   Nơi chứa toàn bộ logic xử lý nghiệp vụ chính, tải mô hình machine learning (`.joblib`), tính toán, tiền xử lý dữ liệu.
    *   Lớp Service tuyệt đối không phụ thuộc vào FastAPI request/response để dễ dàng viết Unit Test độc lập.
*   **Routes (`backend/app/routes/`)**:
    *   Chỉ làm nhiệm vụ tiếp nhận HTTP request, gọi hàm tương ứng từ Service, bắt lỗi ngoại lệ và trả về Response Model tương ứng.

### 🔗 2. Đăng ký Router mới vào Hệ thống
Sau khi xây dựng xong Router (ví dụ: `predict_route.py`), hãy import và đăng ký vào ứng dụng FastAPI chính tại file [backend/main.py](file:///c:/Users/Admin/Documents/viet_code/python/NMKHDL/customer_churn_prediction/backend/main.py) bằng lệnh:
```python
from app.routes import predict_route
app.include_router(predict_route.router)
```

### 📝 3. Quy chuẩn ghi Log (Logging)
*   **Tuyệt đối không sử dụng hàm `print()`** trong mã nguồn vì nó làm chậm server và khó theo dõi khi deploy.
*   Hãy dùng module `logging` chuẩn của Python:
    ```python
    import logging
    logger = logging.getLogger(__name__)
    
    # Sử dụng trong code
    logger.info("Đã tải thành công mô hình dự đoán churn.")
    logger.error("Lỗi khi đọc file tiền xử lý dữ liệu: %s", str(e))
    ```

### 🧪 4. Viết Unit Test tự động
*   Mọi API mới được viết ra phải đi kèm ít nhất 3-5 kịch bản kiểm thử (Test cases) đặt trong thư mục `backend/tests/`.
*   Đặt tên file kiểm thử bắt đầu bằng `test_*.py` (ví dụ: `test_predict.py`).
*   Chạy thử bộ test case cục bộ trước khi push code lên Git bằng lệnh:
    ```bash
    pytest backend/tests/ -v
    ```
