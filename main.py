from src.utils.logger import logger
from src.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline

STAGE_NAME = "Giai đoạn Data Ingestion"

try:
    logger.info(f">>>>>> Bắt đầu giai đoạn {STAGE_NAME} <<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> Hoàn thành giai đoạn {STAGE_NAME} <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e
