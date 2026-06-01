import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.routes import eda_route

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (line %(lineno)d): %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Backend API phục vụ cho ứng dụng dự đoán khách hàng rời bỏ dịch vụ.",
    version="1.0.0"
)

# Cấu hình CORS để Frontend (HTML/JS) gọi được API từ cổng khác
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Trong thực tế nên giới hạn domain cụ thể
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Đăng ký các router
app.include_router(eda_route.router, prefix="/api/v1")

@app.get("/")
def read_root():
    return {"message": "Chào mừng đến với API dự đoán rời bỏ dịch vụ khách hàng!"}
