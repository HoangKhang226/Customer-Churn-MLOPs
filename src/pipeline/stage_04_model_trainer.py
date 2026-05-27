from src.config.configuration import ConfigurationManager
from src.components.model_trainer import ModelTrainer
from src.utils.logger import logger

STAGE_NAME = "Giai đoạn Model Training"


class ModelTrainerTrainingPipeline:
    """
    Pipeline thực thi giai đoạn Model Training (Huấn luyện mô hình).
    """
    def __init__(self):
        """Khởi tạo pipeline cho Model Training."""
        pass

    def main(self):
        """
        Luồng chạy chính của Model Training.
        1. Khởi tạo ConfigurationManager để đọc các thông tin từ config.yaml và params.yaml.
        2. Lấy cấu hình riêng cho bước model training.
        3. Truyền cấu hình vào component ModelTrainer.
        4. Gọi hàm initiate_model_trainer để tiến hành huấn luyện mô hình.
        """
        try:
            config = ConfigurationManager()
            model_trainer_config = config.get_model_trainer_config()
            model_trainer = ModelTrainer(config=model_trainer_config)
            model_trainer.initiate_model_trainer()
        except Exception as e:
            raise e


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> Bắt đầu giai đoạn {STAGE_NAME} <<<<<<")
        obj = ModelTrainerTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> Hoàn thành giai đoạn {STAGE_NAME} <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
