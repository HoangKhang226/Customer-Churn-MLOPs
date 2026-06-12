import logging

from fastapi import APIRouter, HTTPException, Query

from app.config import TRAIN_DATA_PATH
from app.schemas.eda_schema import (
    BivariateAnalysisResponse,
    CategoricalDistributionResponse,
    CorrelationMatrixResponse,
    DatasetOverviewResponse,
    NumericalDistributionResponse,
    NumericalStatsResponse,
    SanityCheckResponse,
)
from app.services.eda_service import EDAService


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/eda",
    tags=["Exploratory Data Analysis"],
)

try:
    eda_service = EDAService(TRAIN_DATA_PATH)
except Exception as exc:
    logger.error("Khong the khoi tao EDAService tai router: %s", exc)
    eda_service = None


def verify_service():
    if eda_service is None:
        raise HTTPException(
            status_code=500,
            detail="EDA Service chua duoc khoi tao thanh cong do thieu du lieu hoac loi nap file.",
        )


@router.get("/overview", response_model=DatasetOverviewResponse)
def get_overview():
    verify_service()
    try:
        return eda_service.get_dataset_overview()
    except Exception as exc:
        logger.error("Loi endpoint /overview: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/sanity-check", response_model=SanityCheckResponse)
def get_sanity_check():
    verify_service()
    try:
        return eda_service.get_data_sanity_check()
    except Exception as exc:
        logger.error("Loi endpoint /sanity-check: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/numerical-stats", response_model=NumericalStatsResponse)
def get_numerical_stats():
    verify_service()
    try:
        return eda_service.get_numerical_statistics()
    except Exception as exc:
        logger.error("Loi endpoint /numerical-stats: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/distribution/numerical/{column_name}", response_model=NumericalDistributionResponse)
def get_numerical_distribution(column_name: str, bins: int = Query(15, ge=5, le=50)):
    verify_service()
    try:
        return eda_service.get_numerical_distribution(column_name, bins)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Loi endpoint /distribution/numerical/%s: %s", column_name, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/distribution/categorical/{column_name}", response_model=CategoricalDistributionResponse)
def get_categorical_distribution(column_name: str):
    verify_service()
    try:
        return eda_service.get_categorical_distribution(column_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Loi endpoint /distribution/categorical/%s: %s", column_name, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/bivariate/{feature_name}", response_model=BivariateAnalysisResponse)
def get_bivariate_analysis(feature_name: str):
    verify_service()
    try:
        return eda_service.get_bivariate_analysis(feature_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Loi endpoint /bivariate/%s: %s", feature_name, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/correlation", response_model=CorrelationMatrixResponse)
def get_correlation():
    verify_service()
    try:
        return eda_service.get_correlation_matrix()
    except Exception as exc:
        logger.error("Loi endpoint /correlation: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
