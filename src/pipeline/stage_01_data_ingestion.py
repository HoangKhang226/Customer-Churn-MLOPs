from src.config.configuration import ConfigurationManager
from src.components.data_ingestion import DataIngestion
from src.utils.logger import logger

STAGE_NAME = "Giai đoạn Data Ingestion"

class DataIngestionTrainingPipeline:
    """
    Pipeline thực thi giai đoạn Data Ingestion (Tập hợp và đưa dữ liệu vào hệ thống).
    """
    def __init__(self):
        """Khởi tạo pipeline cho Data Ingestion."""
        pass

    def main(self):
        """
        Luồng chạy chính của Data Ingestion.
        1. Khởi tạo ConfigurationManager để đọc các thông tin từ config.yaml.
        2. Lấy cấu hình riêng cho bước data ingestion.
        3. Truyền cấu hình vào component DataIngestion.
        4. Gọi hàm extract_zip_file để tiến hành giải nén dữ liệu.
        """
        config = ConfigurationManager() # 1. Lấy tất cả cấu hình
        data_ingestion_config = config.get_data_ingestion_config() # 2. Lấy cấu hình cho Data Ingestion
        data_ingestion = DataIngestion(config=data_ingestion_config) # 3. Khởi tạo component
        data_ingestion.extract_zip_file() # 4. Bắt đầu giải nén file data zip

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> Bắt đầu giai đoạn {STAGE_NAME} <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> Hoàn thành giai đoạn {STAGE_NAME} <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
