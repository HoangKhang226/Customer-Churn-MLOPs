from src.utils.common import read_yaml, create_directories
from src.entity.config_entity import DataIngestionConfig
from pathlib import Path
import os

CONFIG_FILE_PATH = Path("config/config.yaml")

class ConfigurationManager:
    """
    Class quản lý các cấu hình của project.
    Có nhiệm vụ đọc file yaml và khởi tạo các object chứa thông tin đường dẫn tương ứng cho từng bước trong pipeline.
    """
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH):
        """
        Khởi tạo ConfigurationManager.
        
        Args:
            config_filepath (Path): Đường dẫn mặc định đến file config.yaml.
        """
        self.config = read_yaml(config_filepath) # đọc file config.yaml
        create_directories([self.config.artifacts_root]) # tạo folder artifacts gốc

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Lấy thông tin cấu hình cho bước Data Ingestion.
        
        Returns:
            DataIngestionConfig: Object chứa các đường dẫn cần thiết (root_dir, local_data_file, unzip_dir).
        """
        config = self.config.data_ingestion

        # Tạo thư mục chứa dữ liệu đầu vào (data_ingestion) trong artifacts
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir)
        ) # thêm các đường dẫn vào object

        return data_ingestion_config
