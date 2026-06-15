from fastapi import FastAPI
from src.backend.endpoints import predict

app = FastAPI(title="Churn Prediction API", version="1.0")

# Register routes
app.include_router(predict.router)

@app.get("/")
def health_check():
    return {"status": "ok", "message": "FastAPI is running"}
