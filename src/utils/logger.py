import sys

if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

import logging
import logging.config
from pathlib import Path
import yaml

# Xác định đường dẫn gốc và file cấu hình logging
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_LOG_FILE = _PROJECT_ROOT / "config" / "logging.yaml"
_LOGGING_DIR = _PROJECT_ROOT / "logs"

def setup_logging(config_path: Path = _LOG_FILE) -> None:
    """
    Cấu hình logging từ file YAML.
    
    Tự động tạo thư mục 'logs' nếu chưa có và áp dụng cấu hình từ file YAML.
    
    Args:
        config_path: Đường dẫn đến file cấu hình logging.yaml.
    """
    if not config_path.exists():
        # Fallback cơ bản nếu không tìm thấy file cấu hình
        logging.basicConfig(level=logging.INFO)
        logging.warning(f"Không tìm thấy file {config_path.name}, sử dụng cấu hình mặc định.")
        return

    # Tạo thư mục chứa log nếu chưa tồn tại
    if not _LOGGING_DIR.exists():
        _LOGGING_DIR.mkdir(parents=True, exist_ok=True)

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            logging_config = yaml.safe_load(f)
        logging.config.dictConfig(logging_config)
    except Exception as e:
        logging.basicConfig(level=logging.INFO)
        logging.error(f"Lỗi khi thiết lập logging: {e}")

# Khởi tạo logging khi module được import
setup_logging()

# Logger dùng chung cho toàn bộ
logger = logging.getLogger("customer_churn_prediction")

logger.info("Logger đã được khởi tạo thành công")