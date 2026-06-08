from fastapi.testclient import TestClient

from app.main import app
from app.routes import predict_route


client = TestClient(app)


# Payload mẫu hợp lệ dùng chung cho các test Prediction API.
VALID_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 89.5,
    "TotalCharges": 1074.0,
}


def test_predict_returns_prediction(monkeypatch):
    # Mock service để test contract API mà không cần model.joblib/preprocessor.joblib thật.
    def fake_predict_churn(customer_data):
        return {"churn_prediction": "Yes", "churn_probability": 0.73}

    monkeypatch.setattr(predict_route.predict_service, "predict_churn", fake_predict_churn)

    response = client.post("/api/predict", json=VALID_CUSTOMER)

    assert response.status_code == 200
    body = response.json()
    assert body["churn_prediction"] in ["Yes", "No"]
    assert 0.0 <= body["churn_probability"] <= 1.0


def test_predict_endpoint_matches_backend_plan(monkeypatch):
    # Endpoint /api/predict khớp đúng quy ước trong backend_plan.md.
    def fake_predict_churn(customer_data):
        return {"churn_prediction": "No", "churn_probability": 0.24}

    monkeypatch.setattr(predict_route.predict_service, "predict_churn", fake_predict_churn)

    response = client.post("/api/predict", json=VALID_CUSTOMER)

    assert response.status_code == 200
    assert response.json()["churn_prediction"] == "No"


def test_predict_rejects_missing_required_field():
    payload = dict(VALID_CUSTOMER)
    payload.pop("tenure")

    response = client.post("/api/predict", json=payload)

    assert response.status_code == 422


def test_predict_rejects_invalid_type():
    payload = dict(VALID_CUSTOMER)
    payload["tenure"] = "twelve"

    response = client.post("/api/predict", json=payload)

    assert response.status_code == 422


def test_predict_returns_503_when_artifacts_missing(monkeypatch):
    def fake_predict_churn(customer_data):
        raise FileNotFoundError("Prediction artifacts are missing")

    monkeypatch.setattr(predict_route.predict_service, "predict_churn", fake_predict_churn)

    response = client.post("/api/predict", json=VALID_CUSTOMER)

    assert response.status_code == 503
