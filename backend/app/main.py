import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import eda_route, predict_route

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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Đăng ký router theo quy ước chung: prefix được khai báo trong từng route file.
app.include_router(eda_route.router)
app.include_router(predict_route.router)

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Churn Prediction API is running"}
