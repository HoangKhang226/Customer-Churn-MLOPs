from src.config.configuration import ConfigurationManager
from src.components.data_validation import DataValidation
from src.utils.logger import logger

STAGE_NAME = "Giai đoạn Data Validation"

class DataValidationTrainingPipeline:
    """
    Pipeline thực thi giai đoạn Data Validation (Kiểm tra dữ liệu).
    """
    def __init__(self):
        """Khởi tạo pipeline cho Data Validation."""
        pass

    def main(self):
        """
        Luồng chạy chính của Data Validation.
        1. Khởi tạo ConfigurationManager để đọc các thông tin từ config.yaml.
        2. Lấy cấu hình riêng cho bước data validation.
        3. Truyền cấu hình vào component DataValidation.
        4. Gọi hàm validate_all_columns để kiểm tra dữ liệu.
        """
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        data_validation.validate_all_columns()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> Bắt đầu giai đoạn {STAGE_NAME} <<<<<<")
        obj = DataValidationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> Hoàn thành giai đoạn {STAGE_NAME} <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
