
## Hướng Dẫn Cài Đặt

### 0. Repo
- Lần đâu thì clone repo này về
- Các lần tiếp theo thì sử dụng lệnh pull để lấy các thay đổi mới nhất từ github về
- Làm xong nhớ commit và push lên lại

### 1. Tạo Môi Trường Ảo (Virtual Environment)


#### Trên Windows (PowerShell):
```powershell
# Tạo môi trường ảo tên "venv"
python -m venv venv

# Kích hoạt môi trường ảo
.\venv\Scripts\Activate.ps1
```

#### Trên macOS/Linux:
```bash
# Tạo môi trường ảo tên "venv"
python3 -m venv venv

# Kích hoạt môi trường ảo
source venv/bin/activate
```

### 2. Cài Đặt Các Thư Viện Cần Thiết

**Sau khi kích hoạt môi trường ảo**, chạy lệnh:
```bash
pip install -r requirements.txt
```
### 3. Xác Nhận Cài Đặt

Để kiểm tra xem các thư viện đã cài đặt thành công chưa:
```bash
pip list
```

## Bắt đầu

### Đăng ký kernel
```bash
python -m ipykernel install --user --name venv_kernel --display-name "Python (venv)"
```
 Kích hoạt kernel bằng select kernel trong fiel .ipynb

 ### Tạo thư mục data/
 - Tạo thư mục tên data
 - Tải 3 files data từ kaggle về : https://www.kaggle.com/competitions/playground-series-s6e3/data
 - Lưu 3 files đó vào thư mục data
