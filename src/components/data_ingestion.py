import os
import zipfile
from pathlib import Path
from src.utils.logger import logger
from src.entity.config_entity import DataIngestionConfig

class DataIngestion:
    """
    Component chịu trách nhiệm xử lý việc đưa dữ liệu (Data Ingestion) vào hệ thống.
    Đọc dữ liệu thô từ file zip và giải nén vào thư mục artifacts.
    """
    def __init__(self, config: DataIngestionConfig):
        """
        Khởi tạo class DataIngestion.

        Args:
            config (DataIngestionConfig): Đối tượng chứa cấu hình cho Data Ingestion.
        """
        self.config = config

    def extract_zip_file(self):
        """
        Giải nén file dữ liệu định dạng .zip vào thư mục được chỉ định trong cấu hình.
        """
        unzip_path = self.config.unzip_dir
        
        # Tạo thư mục giải nén nếu chưa tồn tại
        os.makedirs(unzip_path, exist_ok=True)
        
        local_file = str(self.config.local_data_file)
        
        # Kiểm tra xem file nguồn có tồn tại hay không
        if not os.path.exists(local_file):
            logger.error(f"Không tìm thấy file: {local_file}")
            raise FileNotFoundError(f"Không tìm thấy file: {local_file}")

        # Tiến hành giải nén nếu là file zip
        if local_file.endswith(".zip"):
            try:
                with zipfile.ZipFile(local_file, "r") as zip_ref:
                    zip_ref.extractall(unzip_path)
                logger.info(f"Đã giải nén thành công {local_file} vào thư mục {unzip_path}")
            except Exception as e:
                logger.error(f"Giải nén file {local_file} thất bại: {e}")
                raise e
        else:
            logger.warning(f"File đầu vào mong đợi là .zip, nhưng nhận được: {local_file}")
