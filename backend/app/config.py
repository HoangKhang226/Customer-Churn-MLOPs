from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

# Quy ước chung trong backend_plan.md: các file artifact phục vụ backend nằm trong app/ml_artifacts.
ML_ARTIFACTS_DIR = BASE_DIR / "ml_artifacts"
DATA_DIR = BASE_DIR.parent / "data"

MODEL_PATH = ML_ARTIFACTS_DIR / "model.joblib"
PREPROCESSOR_PATH = ML_ARTIFACTS_DIR / "preprocessor.joblib"
TRAIN_DATA_PATH = DATA_DIR / "train.csv"
