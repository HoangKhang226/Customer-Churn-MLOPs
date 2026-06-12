# 📊 Telco Customer Churn — Exploratory Data Analysis

Notebook thực hiện toàn bộ quy trình EDA và Feature Engineering trên tập dữ liệu Telco Churn, chuẩn bị đặc trưng đầu vào cho pipeline MLOps.

---

## 📁 Cấu trúc Notebook

### 0. Environment Setup
Khởi tạo môi trường: import thư viện, cấu hình hiển thị, tải dữ liệu thô.

---

### 1. Data Preprocessing
Tiền xử lý cơ bản: chuẩn hóa kiểu dữ liệu, xử lý `TotalCharges` bị lưu dạng chuỗi.

---

### 2. Dataset Overview
- Mô tả sơ bộ kích thước, kiểu dữ liệu, tỷ lệ missing values.
- Phân loại vai trò từng biến: định lượng / định tính / mục tiêu.

---

### 3. Data Quality Assessment
- **Numerical Sanity Check**: kiểm tra khoảng giá trị hợp lệ, phát hiện bất thường.
- **Outlier Detection**: khảo sát phân phối, xác định và kiểm soát điểm ngoại lai.
- **Categorical Sanity Check**: kiểm tra tính nhất quán logic chéo giữa các biến dịch vụ (ví dụ: không thể dùng dịch vụ Internet nếu không có Internet), phát hiện và loại bỏ 7 ghost customers không đăng ký bất kỳ dịch vụ nào.

---

### 4. Univariate Analysis
- **Numerical Features**: phân phối `tenure`, `MonthlyCharges`, `TotalCharges`.
- **Categorical Features**: tần suất từng nhãn trong các biến dịch vụ và hành vi.
- **Target Variable**: phân bố lớp Churn — mất cân bằng 77.5% No / 22.5% Yes.

---

### 5. Bivariate Analysis & Feature Engineering

#### 5.1 Numerical Features vs Target
Phân tích phân phối KDE và boxplot của các biến định lượng theo nhãn Churn. Phát hiện hiện tượng Early Churn tập trung ở `tenure` thấp.

#### 5.2 Categorical Features vs Target
- Xây dựng bảng **Risk Spread** đo khoảng phân hóa tỷ lệ rời bỏ giữa các nhãn trong từng đặc trưng.
- Xây dựng bảng **Tỷ lệ rời bỏ dự kiến** (Quy mô × Tỷ lệ rời bỏ) để định vị vùng rủi ro thực sự, tránh bẫy thống kê từ các phân khúc nhỏ tỷ lệ cao.
- **Gap Analysis**: định lượng sức giữ chân của từng dịch vụ bổ trợ.

#### 5.3 Feature Engineering
Tạo 9 đặc trưng phái sinh từ insight EDA:

| Đặc trưng | Loại | Mô tả |
|:---|:---|:---|
| `loyalty_tier` | Phân khúc | Phân khúc thời gian gắn bó (`tenure`) |
| `charge_segment` | Phân khúc | Phân khúc cước phí hàng tháng |
| `total_active_services` | Đếm | Tổng số dịch vụ đang sử dụng |
| `charge_to_tenure_ratio_log` | Tài chính | Log áp lực chi phí trên thời gian gắn bó |
| `average_cost_per_service` | Tài chính | Đơn giá trung bình mỗi dịch vụ |
| `security_score` | Điểm số | Điểm Khiên Bảo vệ [-1, 0..4], -1 = không có Internet |
| `streaming_score` | Điểm số | Điểm Giải trí [-1, 0..2], -1 = không có Internet |
| `manual_payment` | Cờ | Thanh toán thủ công (Electronic/Mailed check) |
| `composite_risk_profile` | Siêu cờ | Month-to-month AND Fiber optic |
| `demographic_profile` | Biến nén | Hồ sơ nhân khẩu học từ SeniorCitizen + Partner + Dependents |

#### 5.4 Global Correlation Analysis
- Pearson / Spearman cho biến định lượng & thứ bậc.
- Cramér's V cho biến định tính & điểm số.
- Bảng xếp hạng sức mạnh đặc trưng tổng hợp.

#### 5.5 Feature Selection
- Liệt kê đặc trưng được giữ lại và lý do loại bỏ.
- Đóng gói schema cuối cùng chuyển giao cho pipeline.

---

### 6. Pipeline Handover
Tổng kết toàn bộ đặc trưng tối ưu, giải thích ranh giới trách nhiệm giữa EDA notebook và MLOps pipeline (encoding, scaling, SMOTE được thực hiện hoàn toàn ở pipeline).

---

## 🔗 Đặc trưng đầu ra chuyển giao Pipeline

Xem chi tiết schema đầy đủ gồm loại biến, mô tả và công thức tính tại mục **5.5.3** trong notebook.

---

## ⚠️ Lưu ý cho Pipeline

- Các cột gốc như `OnlineSecurity`, `TechSupport`,... **chưa được encode** — pipeline chịu trách nhiệm toàn bộ encoding.
- `security_score` và `streaming_score` mang giá trị **-1** cho nhóm không có Internet — cần xử lý như biến thứ bậc, không phải nhị phân.
- `average_cost_per_service` đã được đảm bảo không có NaN hay Inf (kiểm tra tại mục 3.3).
