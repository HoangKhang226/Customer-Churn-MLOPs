import logging
from functools import lru_cache
from math import isfinite
from pathlib import Path
from typing import Any, Dict, Tuple
import joblib
import numpy as np
import pandas as pd
from app.config import MODEL_PATH, PREPROCESSOR_PATH


logger = logging.getLogger(__name__)


# Load model và preprocessor một lần rồi cache lại để các request sau không phải đọc file lại.
@lru_cache(maxsize=1)
def load_prediction_artifacts() -> Tuple[Any, Any]:
    missing_paths = [
        path.name
        for path in (MODEL_PATH, PREPROCESSOR_PATH)
        if not Path(path).exists()
    ]
    if missing_paths:
        raise FileNotFoundError(
            "Thiếu file mô hình hoặc bộ tiền xử lý: " + ", ".join(missing_paths)
        )

    logger.info("Đang tải mô hình dự đoán từ %s", MODEL_PATH)
    model = joblib.load(MODEL_PATH)

    logger.info("Đang tải bộ tiền xử lý từ %s", PREPROCESSOR_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    return model, preprocessor


def _build_customer_frame(customer_data: Dict[str, Any]) -> pd.DataFrame:
    data = dict(customer_data)

    # Các cột này đã bị loại trong pipeline huấn luyện nên không đưa vào preprocessor.
    for column in ("id", "TotalCharges", "gender"):
        data.pop(column, None)
    return pd.DataFrame([data])


def _get_churn_probability(model: Any, transformed_customer: Any) -> float:
    if not hasattr(model, "predict_proba"):
        raw_prediction = model.predict(transformed_customer)[0]
        normalized_prediction = _normalize_prediction_label(raw_prediction)
        return 1.0 if normalized_prediction == "Yes" else 0.0

    probabilities = model.predict_proba(transformed_customer)[0]
    class_labels = getattr(model, "classes_", None)

    if class_labels is not None and 1 in class_labels:
        churn_index = list(class_labels).index(1)
    elif class_labels is not None and "Yes" in class_labels:
        churn_index = list(class_labels).index("Yes")
    elif len(probabilities) == 2:
        churn_index = 1
    else:
        raise ValueError("Không xác định được lớp Churn trong kết quả dự đoán của mô hình")

    churn_probability = float(probabilities[churn_index])
    if not isfinite(churn_probability) or not 0.0 <= churn_probability <= 1.0:
        raise ValueError("Xác suất dự đoán không hợp lệ")

    return churn_probability


def _normalize_prediction_label(raw_prediction: Any) -> str:
    if isinstance(raw_prediction, str):
        return "Yes" if raw_prediction.lower() == "yes" else "No"

    if isinstance(raw_prediction, (int, float, np.integer, np.floating)):
        return "Yes" if int(raw_prediction) == 1 else "No"

    raise ValueError("Nhãn dự đoán của mô hình không hợp lệ")


# Hàm nghiệp vụ chính: nhận dữ liệu khách hàng, tiền xử lý, chạy model và trả nhãn + xác suất churn.
def predict_churn(customer_data: Dict[str, Any]) -> Dict[str, Any]:
    model, preprocessor = load_prediction_artifacts()
    customer_df = _build_customer_frame(customer_data)

    transformed = preprocessor.transform(customer_df)
    churn_probability = _get_churn_probability(model, transformed)
    churn_prediction = "Yes" if churn_probability >= 0.5 else "No"

    return {
        "churn_prediction": churn_prediction,
        "churn_probability": churn_probability,
    }
