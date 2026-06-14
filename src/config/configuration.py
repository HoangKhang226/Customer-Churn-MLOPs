from src.utils.common import read_yaml, create_directories
from src.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    PredictionConfig,
)
from pathlib import Path
import os

CONFIG_FILE_PATH = Path("config/config.yaml")
SCHEMA_FILE_PATH = Path("config/schema.yaml")
PARAMS_FILE_PATH = Path("config/params.yaml")

class ConfigurationManager:
    """
    Class quản lý các cấu hình của project.
    Có nhiệm vụ đọc file yaml và khởi tạo các object chứa thông tin đường dẫn tương ứng cho từng bước trong pipeline.
    """
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):
        """
        Khởi tạo ConfigurationManager.
        
        Args:
            config_filepath (Path): Đường dẫn mặc định đến file config.yaml.
            schema_filepath (Path): Đường dẫn mặc định đến file schema.yaml.
            params_filepath (Path): Đường dẫn mặc định đến file params.yaml.
        """
        self.config = read_yaml(config_filepath)
        self.schema = read_yaml(schema_filepath)
        self.params = read_yaml(params_filepath)
        
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Lấy thông tin cấu hình cho bước Data Ingestion.
        """
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir)
        )

        return data_ingestion_config

    def get_data_validation_config(self) -> DataValidationConfig:
        """
        Lấy thông tin cấu hình cho bước Data Validation.
        """
        config = self.config.data_validation
        schema = self.schema.COLUMNS

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=Path(config.root_dir),
            STATUS_FILE=config.STATUS_FILE,
            unzip_data_dir=Path(config.unzip_data_dir),
            all_schema=schema,
        )

        return data_validation_config

    def get_data_transformation_config(self) -> DataTransformationConfig:
        """
        Lấy thông tin cấu hình cho bước Data Transformation.
        """
        config = self.config.data_transformation
        create_directories([config.root_dir])
        return DataTransformationConfig(
            root_dir=Path(config.root_dir),
            train_data_path=Path(config.train_data_path),
            test_data_path=Path(config.test_data_path),
            preprocessor_path=Path(config.preprocessor_path),
        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        """
        Lấy thông tin cấu hình cho bước Model Trainer.
        """
        config = self.config.model_trainer
        params = self.params

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            train_data_path=Path(config.train_data_path),
            test_data_path=Path(config.test_data_path),
            model_name=config.model_name,
            lgbm_params=params.LightGBM,
            xgboost_params=params.XGBoost,
            catboost_params=params.CatBoost,
            mlflow_uri=config.mlflow_uri,
        )

        return model_trainer_config

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        """
        Lấy thông tin cấu hình cho bước Model Evaluation.
        """
        config = self.config.model_evaluation
        params = self.params

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=Path(config.root_dir),
            test_data_path=Path(config.test_data_path),
            model_path=Path(config.model_path),
            all_params=params,
            metric_file_name=Path(config.metric_file_name),
            mlflow_uri=config.mlflow_uri,
        )

        return model_evaluation_config

    def get_prediction_config(self) -> PredictionConfig:
        """
        Lấy thông tin cấu hình cho bước Prediction & Submission.
        """
        config = self.config.prediction

        create_directories([config.root_dir])

        prediction_config = PredictionConfig(
            root_dir=Path(config.root_dir),
            model_path=Path(config.model_path),
            preprocessor_path=Path(config.preprocessor_path),
            test_data_path=Path(config.test_data_path),
            output_path=Path(config.output_path),
        )

        return prediction_config
