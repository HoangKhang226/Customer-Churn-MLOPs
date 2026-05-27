from src.utils.logger import logger
from src.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from src.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from src.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline
from src.pipeline.stage_05_model_evaluation import ModelEvaluationTrainingPipeline

STAGE_NAME = "Giai đoạn Data Ingestion"
try:
    logger.info(f">>>>>> Bắt đầu giai đoạn {STAGE_NAME} <<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> Hoàn thành giai đoạn {STAGE_NAME} <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Giai đoạn Data Validation"
try:
    logger.info(f">>>>>> Bắt đầu giai đoạn {STAGE_NAME} <<<<<<")
    data_validation = DataValidationTrainingPipeline()
    data_validation.main()
    logger.info(f">>>>>> Hoàn thành giai đoạn {STAGE_NAME} <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Giai đoạn Data Transformation"
try:
    logger.info(f">>>>>> Bắt đầu giai đoạn {STAGE_NAME} <<<<<<")
    data_transformation = DataTransformationTrainingPipeline()
    data_transformation.main()
    logger.info(f">>>>>> Hoàn thành giai đoạn {STAGE_NAME} <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Giai đoạn Model Training"
try:
    logger.info(f">>>>>> Bắt đầu giai đoạn {STAGE_NAME} <<<<<<")
    model_trainer = ModelTrainerTrainingPipeline()
    model_trainer.main()
    logger.info(f">>>>>> Hoàn thành giai đoạn {STAGE_NAME} <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Giai đoạn Model Evaluation"
try:
    logger.info(f">>>>>> Bắt đầu giai đoạn {STAGE_NAME} <<<<<<")
    model_evaluation = ModelEvaluationTrainingPipeline()
    model_evaluation.main()
    logger.info(f">>>>>> Hoàn thành giai đoạn {STAGE_NAME} <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

