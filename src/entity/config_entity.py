from dataclasses import dataclass
from pathlib import Path
from pickle import TRUE

"""
cấu hình cho các bước trong pipeline
"""

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    unzip_data_dir: Path
    all_schema: dict

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    preprocessor_path: Path

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_name: str
    lgbm_params: dict
    xgboost_params: dict
    catboost_params: dict
    mlflow_uri: str

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    all_params: dict
    metric_file_name: Path
    mlflow_uri: str

@dataclass(frozen=True)
class PredictionConfig:
    root_dir: Path
    model_path: Path
    preprocessor_path: Path
    test_data_path: Path
    output_path: Path