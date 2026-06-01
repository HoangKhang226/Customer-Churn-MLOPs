from pydantic import BaseModel, Field
from typing import Dict, List, Any, Union

# --- 1. Schema cho get_dataset_overview ---
class DatasetShape(BaseModel):
    rows: int = Field(..., description="Số lượng dòng trong tập dữ liệu")
    columns: int = Field(..., description="Số lượng cột trong tập dữ liệu")

class FeatureRoles(BaseModel):
    identifiers: List[str] = Field(..., description="Danh sách cột định danh (như id)")
    numerical: List[str] = Field(..., description="Danh sách các cột chứa giá trị số")
    categorical: List[str] = Field(..., description="Danh sách các cột phân loại")
    target: List[str] = Field(..., description="Cột mục tiêu cần dự đoán (Churn)")

class DatasetOverviewResponse(BaseModel):
    shape: DatasetShape = Field(..., description="Kích thước tập dữ liệu")
    duplicates: int = Field(..., description="Số dòng bị trùng lặp dữ liệu")
    missing_values_count: int = Field(..., description="Tổng số lượng giá trị bị trống (NULL/NaN)")
    column_types: Dict[str, str] = Field(..., description="Kiểu dữ liệu thực tế của từng cột")
    feature_roles: FeatureRoles = Field(..., description="Phân vai trò nghiệp vụ của các cột")
    insight: str = Field(..., description="Đoạn văn nhận xét phân tích tổng quan")


# --- 2. Schema cho get_data_sanity_check ---
class NumericalSanity(BaseModel):
    tenure_invalid: int = Field(..., description="Số lượng dòng bị lỗi tenure <= 0 hoặc lẻ")
    monthly_charges_invalid: int = Field(..., description="Số lượng dòng lỗi MonthlyCharges <= 0")
    total_charges_invalid: int = Field(..., description="Số lượng dòng lỗi TotalCharges <= 0")

class CategoricalSanity(BaseModel):
    internet_logic_errors: int = Field(..., description="Số lượng dòng lỗi đăng ký dịch vụ Internet nhưng InternetService = No")

class SanityCheckResponse(BaseModel):
    numerical_sanity: NumericalSanity = Field(..., description="Lỗi logic các cột dạng số")
    categorical_sanity: CategoricalSanity = Field(..., description="Lỗi logic các cột dạng chữ")
    insight: str = Field(..., description="Đoạn văn nhận xét tính toàn vẹn của dữ liệu")


# --- 3. Schema cho get_numerical_statistics ---
class ColumnStats(BaseModel):
    mean: float = Field(..., description="Giá trị trung bình")
    min: float = Field(..., description="Giá trị nhỏ nhất")
    max: float = Field(..., description="Giá trị lớn nhất")
    skewness: float = Field(..., description="Độ lệch phân phối")
    variance: float = Field(..., description="Phương sai")
    nunique: int = Field(..., description="Số lượng giá trị duy nhất")
    q1: float = Field(..., description="Phân vị 25% (Q1)")
    median: float = Field(..., description="Trung vị / Phân vị 50%")
    q3: float = Field(..., description="Phân vị 75% (Q3)")

class NumericalStatsResponse(BaseModel):
    tenure: ColumnStats = Field(..., description="Thống kê mô tả của cột tenure")
    MonthlyCharges: ColumnStats = Field(..., description="Thống kê mô tả của cột MonthlyCharges")
    TotalCharges: ColumnStats = Field(..., description="Thống kê mô tả của cột TotalCharges")
    insight: str = Field(..., description="Đoạn văn nhận xét phân bố định lượng")


# --- 4. Schema cho get_numerical_distribution ---
class BoxplotData(BaseModel):
    min: float = Field(..., alias="min", description="Giá trị nhỏ nhất của Boxplot")
    q1: float = Field(..., description="Phân vị 25% (Cạnh dưới hộp)")
    median: float = Field(..., description="Trung vị (Đường giữa hộp)")
    q3: float = Field(..., description="Phân vị 75% (Cạnh trên hộp)")
    max: float = Field(..., alias="max", description="Giá trị lớn nhất của Boxplot")

class NumericalDistributionResponse(BaseModel):
    labels: List[str] = Field(..., description="Tên các khoảng chia cột của đồ thị Histogram")
    values: List[int] = Field(..., description="Số lượng dòng rơi vào từng khoảng")
    boxplot_data: BoxplotData = Field(..., description="5 chỉ số để Frontend vẽ đồ thị hộp Boxplot")
    insight: str = Field(..., description="Nhận xét phân phối của cột số này")


# --- 5. Schema cho get_categorical_distribution ---
class CategoricalDistributionResponse(BaseModel):
    labels: List[str] = Field(..., description="Tên các nhóm phân loại")
    counts: List[int] = Field(..., description="Số lượng dòng của từng nhóm")
    percentages: List[float] = Field(..., description="Tỷ lệ % của từng nhóm")
    insight: str = Field(..., description="Nhận xét phân phối của cột phân loại này")


# --- 6. Schema cho get_bivariate_analysis ---
class BivariateCategoricalResponse(BaseModel):
    type: str = Field("categorical", description="Loại phân tích biến phân loại")
    index: List[str] = Field(..., description="Tên các nhóm của biến")
    columns: List[str] = Field(..., description="Nhãn biến mục tiêu Churn (['No', 'Yes'])")
    values: List[List[int]] = Field(..., description="Bảng chéo tần suất của biến với Churn")
    insight: str = Field(..., description="Nhận xét mức độ ảnh hưởng của biến này tới Churn")

class GroupStats(BaseModel):
    mean: float = Field(..., description="Cước phí trung bình của nhóm Churn tương ứng")
    boxplot: List[float] = Field(..., description="5 chỉ số vẽ Boxplot [min, q1, median, q3, max] của nhóm")

class BivariateNumericalResponse(BaseModel):
    type: str = Field("numerical", description="Loại phân tích biến số")
    churn_yes_stats: GroupStats = Field(..., description="Thống kê của nhóm Churn = Yes")
    churn_no_stats: GroupStats = Field(..., description="Thống kê của nhóm Churn = No")
    insight: str = Field(..., description="Nhận xét sự chênh lệch phân phối giữa nhóm đi và ở lại")

# Tổng hợp dạng Union để tự động map theo trường "type"
BivariateAnalysisResponse = Union[BivariateCategoricalResponse, BivariateNumericalResponse]


# --- 7. Schema cho get_correlation_matrix ---
class CorrelationMatrixResponse(BaseModel):
    columns: List[str] = Field(..., description="Danh sách tên các cột số")
    index: List[str] = Field(..., description="Danh sách tên hàng tương ứng")
    values: List[List[float]] = Field(..., description="Mảng 2 chiều chứa ma trận hệ số tương quan")
    insight: str = Field(..., description="Đánh giá tương quan tuyến tính của các biến")
