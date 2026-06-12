import logging

from fastapi import APIRouter, HTTPException

from app.schemas.predict_schema import CustomerInput, PredictionResponse
from app.services import predict_service


logger = logging.getLogger(__name__)

# Router riêng cho phần Người 1: Prediction API.
router = APIRouter(
    prefix="/api",
    tags=["Dự đoán Churn"],
)


@router.post("/predict", response_model=PredictionResponse)
def predict_customer_churn(customer: CustomerInput):
    # Route chỉ nhận request, gọi service xử lý nghiệp vụ, rồi chuẩn hóa lỗi HTTP trả về frontend.
    try:
        return predict_service.predict_churn(customer.model_dump())
    except FileNotFoundError as exc:
        logger.error("Chưa có artifact phục vụ dự đoán: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Chưa có file model.joblib hoặc preprocessor.joblib trong backend/app/ml_artifacts.",
        )
    except Exception as exc:
        logger.exception("Dự đoán thất bại")
        raise HTTPException(status_code=500, detail="Dự đoán thất bại do lỗi xử lý phía server.")
