from src.config.configuration import ConfigurationManager
from src.components.model_evaluation import ModelEvaluation
from src.utils.logger import logger

STAGE_NAME = "Giai đoạn Model Evaluation"


class ModelEvaluationTrainingPipeline:
    """
    Pipeline thực thi giai đoạn Model Evaluation (Đánh giá mô hình).
    """
    def __init__(self):
        """Khởi tạo pipeline cho Model Evaluation."""
        pass

    def main(self):
        """
        Luồng chạy chính của Model Evaluation.
        1. Khởi tạo ConfigurationManager để đọc các thông tin từ config.yaml và params.yaml.
        2. Lấy cấu hình riêng cho bước model evaluation.
        3. Truyền cấu hình vào component ModelEvaluation.
        4. Gọi hàm initiate_model_evaluation để tiến hành đánh giá mô hình.
        """
        try:
            config = ConfigurationManager()
            model_evaluation_config = config.get_model_evaluation_config()
            model_evaluation = ModelEvaluation(config=model_evaluation_config)
            model_evaluation.initiate_model_evaluation()
        except Exception as e:
            raise e


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> Bắt đầu giai đoạn {STAGE_NAME} <<<<<<")
        obj = ModelEvaluationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> Hoàn thành giai đoạn {STAGE_NAME} <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
