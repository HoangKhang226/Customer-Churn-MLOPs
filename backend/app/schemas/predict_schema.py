from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


YesNo = Literal["Yes", "No"]
ServiceStatus = Literal["Yes", "No", "No internet service"]
MultipleLinesStatus = Literal["Yes", "No", "No phone service"]


# Schema dữ liệu đầu vào cho API dự đoán một khách hàng.
# Các Literal giúp FastAPI/Pydantic tự kiểm tra giá trị hợp lệ và sinh Swagger rõ ràng.
class CustomerInput(BaseModel):
    id: Optional[int] = Field(None, description="Mã định danh khách hàng, không bắt buộc khi dự đoán")
    gender: Literal["Male", "Female"] = Field(..., description="Giới tính của khách hàng")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="1 nếu là người cao tuổi, 0 nếu không")
    Partner: YesNo = Field(..., description="Khách hàng có vợ/chồng hoặc bạn đời hay không")
    Dependents: YesNo = Field(..., description="Khách hàng có người phụ thuộc hay không")
    tenure: int = Field(..., ge=0, description="Số tháng khách hàng đã sử dụng dịch vụ")
    PhoneService: YesNo = Field(..., description="Khách hàng có sử dụng dịch vụ điện thoại hay không")
    MultipleLines: MultipleLinesStatus = Field(..., description="Trạng thái sử dụng nhiều đường dây điện thoại")
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(..., description="Loại dịch vụ Internet")
    OnlineSecurity: ServiceStatus = Field(..., description="Trạng thái sử dụng dịch vụ bảo mật trực tuyến")
    OnlineBackup: ServiceStatus = Field(..., description="Trạng thái sử dụng dịch vụ sao lưu trực tuyến")
    DeviceProtection: ServiceStatus = Field(..., description="Trạng thái sử dụng dịch vụ bảo vệ thiết bị")
    TechSupport: ServiceStatus = Field(..., description="Trạng thái sử dụng dịch vụ hỗ trợ kỹ thuật")
    StreamingTV: ServiceStatus = Field(..., description="Trạng thái sử dụng dịch vụ xem TV trực tuyến")
    StreamingMovies: ServiceStatus = Field(..., description="Trạng thái sử dụng dịch vụ xem phim trực tuyến")
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(..., description="Loại hợp đồng")
    PaperlessBilling: YesNo = Field(..., description="Khách hàng có sử dụng hóa đơn điện tử hay không")
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = Field(..., description="Phương thức thanh toán")
    MonthlyCharges: float = Field(..., gt=0, description="Số tiền cước hàng tháng")
    TotalCharges: Optional[float] = Field(None, ge=0, description="Tổng số tiền cước đã tích lũy")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 89.5,
                "TotalCharges": 1074.0,
            }
        }
    )


# Schema dữ liệu trả về cho frontend sau khi model dự đoán.
class PredictionResponse(BaseModel):
    churn_prediction: Literal["Yes", "No"] = Field(..., description="Nhãn dự đoán khách hàng rời bỏ dịch vụ")
    churn_probability: float = Field(..., ge=0.0, le=1.0, description="Xác suất khách hàng rời bỏ dịch vụ")
