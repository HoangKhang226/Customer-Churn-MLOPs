import logging
import json
import sys
import os

# Thêm thư mục gốc vào sys.path để có thể import backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.eda_service import EDAService

# Cấu hình log để xuất ra terminal giống như thực tế
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (line %(lineno)d): %(message)s"
)

# Đường dẫn tới file train.csv trong thư mục data của dự án
DATA_PATH = "data/train.csv"

try:
    print("==================================================")
    print("🔄 KHỞI TẠO SERVICE")
    print("==================================================")
    service = EDAService(DATA_PATH)
    
    # 1. Kiểm tra Dataset Overview
    print("\n==================================================")
    print("📊 1. DATASET OVERVIEW")
    print("==================================================")
    overview = service.get_dataset_overview()
    print(f"Kích thước: {overview['shape']}")
    print(f"Dòng trùng lặp: {overview['duplicates']}")
    print(f"Lớp phân chia đặc trưng: {list(overview['feature_roles'].keys())}")
    print(f"👉 Insight: {overview['insight']}")
    
    # 2. Kiểm tra Sanity Check
    print("\n==================================================")
    print("🔍 2. SANITY CHECK")
    print("==================================================")
    sanity = service.get_data_sanity_check()
    print(f"Lỗi số học định lượng: {sanity['numerical_sanity']}")
    print(f"Lỗi logic Internet: {sanity['categorical_sanity']}")
    print(f"👉 Insight: {sanity['insight']}")

    # # 3. Kiểm tra Outlier Detection
    # print("\n==================================================")
    # print("📦 3. OUTLIER DETECTION")
    # print("==================================================")
    # outliers = service.get_outliers_report()
    # print(f"Ngưỡng và số lượng ngoại lai: {json.dumps(outliers['outliers'], indent=2)}")
    # print(f"👉 Insight: {outliers['insight']}")

    # 4. Kiểm tra phân phối cột số (ví dụ: MonthlyCharges)
    print("\n==================================================")
    print("📈 4. NUMERICAL DISTRIBUTION (MonthlyCharges)")
    print("==================================================")
    dist = service.get_numerical_distribution("MonthlyCharges", bins=5)
    print(f"Nhãn khoảng cước: {dist['labels']}")
    print(f"Số lượng rơi vào từng khoảng: {dist['values']}")
    print(f"👉 Insight: {dist['insight']}")

    # 5. Kiểm tra tương quan đa biến (ví dụ: PaymentMethod vs Churn)
    print("\n==================================================")
    print("🔗 5. BIVARIATE ANALYSIS (PaymentMethod)")
    print("==================================================")
    bivariate = service.get_bivariate_analysis("PaymentMethod")
    print(f"Các phương thức: {bivariate.get('index')}")
    print(f"Bảng tần suất chéo [No, Yes]: {bivariate.get('values')}")
    print(f"👉 Insight: {bivariate.get('insight')}")

except Exception as e:
    print(f"\n❌ Đã xảy ra lỗi: {e}")
