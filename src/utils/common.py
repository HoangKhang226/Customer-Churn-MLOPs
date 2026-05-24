import os
from box.exceptions import BoxValueError
import yaml
from src.utils.logger import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox # convert yaml thành object class
from pathlib import Path
from typing import Any

@ensure_annotations # decorator để ép truyền đúng kiểu dữ liệu
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Đọc file yaml và trả về ConfigBox.

    Args:
        path_to_yaml (Path): Đường dẫn tới file yaml.

    Raises:
        ValueError: Nếu tệp yaml trống.
        e: Lỗi nếu file lỗi.

    Returns:
        ConfigBox: Đối tượng ConfigBox chứa cấu hình.
    """
    try:
        with open(path_to_yaml, encoding="utf-8") as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"Đọc file yaml thành công: {path_to_yaml}")
            if content is None:
                return ConfigBox({})
            return ConfigBox(content)
    except Exception as e:
        logger.error(f"Lỗi khi đọc file yaml {path_to_yaml}: {e}")
        raise e

@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """Tạo ra list các thư mục.

    Args:
        path_to_directories (list): Danh sách đường dẫn của các thư mục cần tạo.
        verbose (bool, tùy chọn): In log khi tạo thư mục. Mặc định là True.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Đã tạo thư mục tại: {path}")

@ensure_annotations
def save_json(path: Path, data: dict):
    """Lưu dữ liệu dưới dạng tệp JSON.

    Args :
        path (Path): Đường dẫn tới tệp JSON.
        data (dict): Dữ liệu sẽ được lưu vào tệp JSON.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    logger.info(f"Đã lưu file JSON tại: {path}")

@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """Tải data từ file JSON.

    Args:
        path (Path): Đường dẫn tới tệp JSON.

    Returns:
        ConfigBox: Data ở dạng class attributes thay vì dict.
    """
    with open(path, encoding="utf-8") as f:
        content = json.load(f)

    logger.info(f"Đã tải file JSON thành công từ: {path}")
    return ConfigBox(content)

@ensure_annotations
def save_bin(data: Any, path: Path):
    """Lưu tệp nhị phân (binary file).

    Args:
        data (Any): Dữ liệu sẽ được lưu dưới dạng nhị phân.
        path (Path): Đường dẫn tới tệp nhị phân.
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"Đã lưu file nhị phân tại: {path}")

@ensure_annotations
def load_bin(path: Path) -> Any:
    """Tải dữ liệu nhị phân.

    Args:
        path (Path): Đường dẫn tới tệp nhị phân.

    Returns:
        Any: Đối tượng được lưu trong file.
    """
    data = joblib.load(path)
    logger.info(f"Đã tải dữ liệu từ file nhị phân: {path}")
    return data

@ensure_annotations
def get_size(path: Path) -> str:
    """Lấy kích thước tệp theo KB.

    Args:
        path (Path): Đường dẫn tới tệp.

    Returns:
        str: Kích thước của tệp tính bằng KB.
    """
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"