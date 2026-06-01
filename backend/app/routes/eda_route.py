import os
import logging
from fastapi import APIRouter, HTTPException, Query
from typing import Any

from backend.app.services.eda_service import EDAService
from backend.app.schemas.eda_schema import (
    DatasetOverviewResponse,
    SanityCheckResponse,
    NumericalStatsResponse,
    NumericalDistributionResponse,
    CategoricalDistributionResponse,
    BivariateAnalysisResponse,
    CorrelationMatrixResponse
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/eda",
    tags=["Exploratory Data Analysis"]
)

# Đường dẫn mặc định tới file dữ liệu huấn luyện
DATA_PATH = os.getenv("EDA_DATA_PATH", "data/train.csv")

try:
    eda_service = EDAService(DATA_PATH)
except Exception as e:
    logger.error(f"Không thể khởi tạo EDAService tại router: {str(e)}")
    eda_service = None

def verify_service():
    if eda_service is None:
        raise HTTPException(
            status_code=500,
            detail="EDA Service chưa được khởi tạo thành công do thiếu dữ liệu hoặc lỗi nạp file."
        )

@router.get("/overview", response_model=DatasetOverviewResponse)
def get_overview():
    """
    [Mục 1 & 2] Lấy thông tin mô tả tổng quan của tập dữ liệu.
    """
    verify_service()
    try:
        return eda_service.get_dataset_overview()
    except Exception as e:
        logger.error(f"Lỗi endpoint /overview: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sanity-check", response_model=SanityCheckResponse)
def get_sanity_check():
    """
    [Mục 3.1 & 3.3] Kiểm tra tính hợp lệ logic nghiệp vụ của các đặc trưng.
    """
    verify_service()
    try:
        return eda_service.get_data_sanity_check()
    except Exception as e:
        logger.error(f"Lỗi endpoint /sanity-check: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/numerical-stats", response_model=NumericalStatsResponse)
def get_numerical_stats():
    """
    [Mục 4.1] Lấy các thống kê đơn biến mô tả của toàn bộ biến định lượng.
    """
    verify_service()
    try:
        return eda_service.get_numerical_statistics()
    except Exception as e:
        logger.error(f"Lỗi endpoint /numerical-stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/distribution/numerical/{column_name}", response_model=NumericalDistributionResponse)
def get_numerical_distribution(column_name: str, bins: int = Query(15, ge=5, le=50)):
    """
    [Mục 4.1] Phân phối tần suất (Histogram & Boxplot) của một biến định lượng cụ thể.
    """
    verify_service()
    try:
        return eda_service.get_numerical_distribution(column_name, bins)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Lỗi endpoint /distribution/numerical/{column_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/distribution/categorical/{column_name}", response_model=CategoricalDistributionResponse)
def get_categorical_distribution(column_name: str):
    """
    [Mục 4.2 & 4.3] Phân phối tần suất (Pie/Bar chart data) của một biến định tính cụ thể.
    """
    verify_service()
    try:
        return eda_service.get_categorical_distribution(column_name)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Lỗi endpoint /distribution/categorical/{column_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/bivariate/{feature_name}", response_model=BivariateAnalysisResponse)
def get_bivariate_analysis(feature_name: str):
    """
    [Phân tích đa biến] Mối quan hệ phân hóa của biến bất kỳ đối với target Churn.
    """
    verify_service()
    try:
        return eda_service.get_bivariate_analysis(feature_name)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Lỗi endpoint /bivariate/{feature_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/correlation", response_model=CorrelationMatrixResponse)
def get_correlation():
    """
    [Mục phân tích tương quan] Ma trận tương quan Pearson giữa các cột số.
    """
    verify_service()
    try:
        return eda_service.get_correlation_matrix()
    except Exception as e:
        logger.error(f"Lỗi endpoint /correlation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
