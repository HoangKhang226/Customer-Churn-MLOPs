import os
import pandas as pd
from src.utils.logger import logger
from src.entity.config_entity import DataValidationConfig

class DataValidation:
    """
    Component chịu trách nhiệm kiểm tra tính hợp lệ của dữ liệu (Data Validation).
    Xác minh xem dữ liệu được ingest có khớp với schema đã định nghĩa hay không.
    """
    def __init__(self, config: DataValidationConfig):
        """
        Khởi tạo class DataValidation.

        Args:
            config (DataValidationConfig): Đối tượng chứa cấu hình cho Data Validation.
        """
        self.config = config

    def validate_all_columns(self) -> bool:
        """
        Kiểm tra xem tất cả các cột trong tập dữ liệu có tồn tại trong schema.yaml hay không.
        Ghi kết quả vào file status.txt.

        Returns:
            bool: Trạng thái hợp lệ của dữ liệu (True nếu hợp lệ, False nếu không).
        """
        try:
            validation_status = None

            data = pd.read_csv(self.config.unzip_data_dir)
            all_cols = list(data.columns)
            all_schema = self.config.all_schema.keys()
            
            # Ghi trạng thái ban đầu
            status_file_path = self.config.STATUS_FILE

            for col in all_cols:
                if col not in all_schema:
                    validation_status = False
                    with open(status_file_path, 'w') as f:
                        f.write(f"Validation status: {validation_status}")
                    logger.error(f"Cột {col} không tồn tại trong schema!")
                    break
                else:
                    validation_status = True

            # Ghi lại kết quả cuối cùng
            with open(status_file_path, 'w') as f:
                f.write(f"Validation status: {validation_status}")
            
            if validation_status:
                logger.info("Tất cả các cột đều hợp lệ so với schema.")
            else:
                logger.warning("Quá trình validate dữ liệu thất bại.")

            return validation_status
        
        except Exception as e:
            logger.error(f"Lỗi trong quá trình Data Validation: {e}")
            raise e
