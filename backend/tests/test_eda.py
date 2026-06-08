from fastapi.testclient import TestClient

from app.main import app
from app.routes import eda_route


client = TestClient(app)


class FakeEDAService:
    def get_dataset_overview(self):
        return {
            "shape": {"rows": 3, "columns": 20},
            "duplicates": 0,
            "missing_values_count": 0,
            "column_types": {"tenure": "int64", "Churn": "object"},
            "feature_roles": {
                "identifiers": ["id"],
                "numerical": ["tenure", "MonthlyCharges", "TotalCharges"],
                "categorical": ["PaymentMethod"],
                "target": ["Churn"],
            },
            "insight": "Dataset overview insight",
        }

    def get_data_sanity_check(self):
        return {
            "numerical_sanity": {
                "tenure_invalid": 0,
                "monthly_charges_invalid": 0,
                "total_charges_invalid": 0,
            },
            "categorical_sanity": {"internet_logic_errors": 0},
            "insight": "Sanity insight",
        }

    def get_numerical_statistics(self):
        stats = {
            "mean": 10.0,
            "min": 1.0,
            "max": 20.0,
            "skewness": 0.1,
            "variance": 2.0,
            "nunique": 3,
            "q1": 5.0,
            "median": 10.0,
            "q3": 15.0,
        }
        return {
            "tenure": stats,
            "MonthlyCharges": stats,
            "TotalCharges": stats,
            "insight": "Numerical stats insight",
        }

    def get_numerical_distribution(self, column_name, bins):
        if column_name == "INVALID_COL":
            raise ValueError("Column INVALID_COL does not exist.")
        return {
            "labels": ["0-10", "10-20"],
            "values": [1, 2],
            "boxplot_data": {"min": 0.0, "q1": 5.0, "median": 10.0, "q3": 15.0, "max": 20.0},
            "insight": "Numerical distribution insight",
        }

    def get_categorical_distribution(self, column_name):
        if column_name == "INVALID_COL":
            raise ValueError("Column INVALID_COL does not exist.")
        return {
            "labels": ["Electronic check", "Credit card (automatic)"],
            "counts": [2, 1],
            "percentages": [66.67, 33.33],
            "insight": "Categorical distribution insight",
        }

    def get_bivariate_analysis(self, feature_name):
        if feature_name == "INVALID_COL":
            raise ValueError("Churn or feature column missing.")
        return {
            "type": "categorical",
            "index": ["Electronic check", "Credit card (automatic)"],
            "columns": ["No", "Yes"],
            "values": [[1, 1], [1, 0]],
            "insight": "Bivariate insight",
        }

    def get_correlation_matrix(self):
        return {
            "columns": ["tenure", "MonthlyCharges", "TotalCharges"],
            "index": ["tenure", "MonthlyCharges", "TotalCharges"],
            "values": [[1.0, 0.2, 0.8], [0.2, 1.0, 0.6], [0.8, 0.6, 1.0]],
            "insight": "Correlation insight",
        }


def test_eda_overview_returns_shape_and_insight(monkeypatch):
    monkeypatch.setattr(eda_route, "eda_service", FakeEDAService())

    response = client.get("/api/v1/eda/overview")

    assert response.status_code == 200
    body = response.json()
    assert body["shape"]["rows"] == 3
    assert body["insight"]


def test_eda_sanity_check_returns_business_logic_errors(monkeypatch):
    monkeypatch.setattr(eda_route, "eda_service", FakeEDAService())

    response = client.get("/api/v1/eda/sanity-check")

    assert response.status_code == 200
    assert "internet_logic_errors" in response.json()["categorical_sanity"]


def test_eda_numerical_stats_returns_three_numeric_columns(monkeypatch):
    monkeypatch.setattr(eda_route, "eda_service", FakeEDAService())

    response = client.get("/api/v1/eda/numerical-stats")

    assert response.status_code == 200
    body = response.json()
    assert {"tenure", "MonthlyCharges", "TotalCharges"}.issubset(body)


def test_eda_numerical_distribution_returns_bins_and_boxplot(monkeypatch):
    monkeypatch.setattr(eda_route, "eda_service", FakeEDAService())

    response = client.get("/api/v1/eda/distribution/numerical/MonthlyCharges")

    assert response.status_code == 200
    body = response.json()
    assert body["labels"]
    assert "boxplot_data" in body


def test_eda_bivariate_returns_cross_tabulation(monkeypatch):
    monkeypatch.setattr(eda_route, "eda_service", FakeEDAService())

    response = client.get("/api/v1/eda/bivariate/PaymentMethod")

    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "categorical"
    assert body["values"]


def test_eda_correlation_returns_square_matrix(monkeypatch):
    monkeypatch.setattr(eda_route, "eda_service", FakeEDAService())

    response = client.get("/api/v1/eda/correlation")

    assert response.status_code == 200
    body = response.json()
    assert len(body["values"]) == len(body["columns"])
    assert all(len(row) == len(body["columns"]) for row in body["values"])


def test_eda_bivariate_rejects_invalid_column(monkeypatch):
    monkeypatch.setattr(eda_route, "eda_service", FakeEDAService())

    response = client.get("/api/v1/eda/bivariate/INVALID_COL")

    assert response.status_code == 400
