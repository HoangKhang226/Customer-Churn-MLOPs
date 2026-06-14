import numpy as np
import os
from src.utils.logger import logger
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import optuna
from catboost import CatBoostClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc,
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import mlflow
import mlflow.sklearn
from src.entity.config_entity import ModelTrainerConfig
import json
from urllib.parse import urlparse


class ModelTrainer:
    """
    Component chịu trách nhiệm huấn luyện mô hình (Model Training).
    Thực hiện Optuna Study trên LightGBM, XGBoost, CatBoost và dùng StackingClassifier.
    """
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def load_transformed_data(self):
        """
        Đọc dữ liệu đã được transform từ file .npz (Stage 03).
        
        Returns:
            tuple: (X_train, y_train, X_test, y_test)
        """
        logger.info(f"Đọc dữ liệu train từ: {self.config.train_data_path}")
        train_data = np.load(self.config.train_data_path, allow_pickle=True)
        X_train = train_data['X'].astype(np.float32)
        y_train = train_data['y'].astype(int)

        logger.info(f"Đọc dữ liệu test từ: {self.config.test_data_path}")
        test_data = np.load(self.config.test_data_path, allow_pickle=True)
        X_test = test_data['X'].astype(np.float32)
        
        # Test data có thể không có nhãn (nếu là tập test của Kaggle)
        if 'y' in test_data.files:
            y_test = test_data['y'].astype(int)
        else:
            y_test = None
            logger.warning("Tập test không có nhãn (y), sẽ chỉ train và validate trên tập train")

        logger.info(f"Kích thước X_train: {X_train.shape}, y_train: {y_train.shape}")
        if y_test is not None:
            logger.info(f"Kích thước X_test: {X_test.shape}, y_test: {y_test.shape}")
        else:
            logger.info(f"Kích thước X_test: {X_test.shape}")

        return X_train, y_train, X_test, y_test

    def hyperparameter_tuning(self, model_name, model_class, param_space, X_train, y_train):
        """
        Thực hiện Optuna Study để tìm siêu tham số tốt nhất.
        
        Args:
            model_name: Tên mô hình ('LightGBM', 'XGBoost', 'CatBoost')
            model_class: Lớp mô hình
            param_space: Không gian tham số Optuna
            X_train: Dữ liệu huấn luyện
            y_train: Nhãn huấn luyện
            
        Returns:
            tuple: (best_estimator, best_params)
        """
        n_trials = param_space.get('n_trials', 10)
        logger.info(f"Bắt đầu Optuna tuning cho {model_name} với {n_trials} trials")
        
        def objective(trial):
            if model_name == "CatBoost":
                params = {
                    'iterations': trial.suggest_int('iterations', param_space['iterations'][0], param_space['iterations'][1]),
                    'depth': trial.suggest_int('depth', param_space['depth'][0], param_space['depth'][1]),
                    'learning_rate': trial.suggest_float('learning_rate', param_space['learning_rate'][0], param_space['learning_rate'][1], log=True)
                }
            else:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', param_space['n_estimators'][0], param_space['n_estimators'][1]),
                    'max_depth': trial.suggest_int('max_depth', param_space['max_depth'][0], param_space['max_depth'][1]),
                    'learning_rate': trial.suggest_float('learning_rate', param_space['learning_rate'][0], param_space['learning_rate'][1], log=True)
                }
                
            if model_name == "LightGBM":
                model = model_class(random_state=42, verbose=-1, **params)
            elif model_name == "CatBoost":
                model = model_class(random_state=42, verbose=0, **params)
            else:
                model = model_class(random_state=42, eval_metric='logloss', n_jobs=-1, **params)
                
            score = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc')
            return score.mean()

        optuna.logging.set_verbosity(optuna.logging.INFO)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        
        logger.info(f"Best params cho {model_name}: {study.best_params}")
        logger.info(f"Best CV ROC AUC score: {study.best_value:.4f}")
        
        best_params = study.best_params
        if model_name == "LightGBM":
            best_model = model_class(random_state=42, verbose=-1, **best_params)
        elif model_name == "CatBoost":
            best_model = model_class(random_state=42, verbose=0, **best_params)
        else:
            best_model = model_class(random_state=42, eval_metric='logloss', n_jobs=-1, **best_params)
            
        return best_model, best_params

    def train_and_log(self, model_name, model, params, X_train, X_val, y_train, y_val):
        """
        Huấn luyện mô hình và log kết quả vào MLflow.
        
        Args:
            model_name: Tên mô hình (LightGBM hoặc XGBoost)
            model: Mô hình đã được tune
            params: Tham số tốt nhất từ Optuna
            X_train, X_val, y_train, y_val: Dữ liệu train và validation
            
        Returns:
            dict: Thông tin mô hình và metrics
        """
        # Thiết lập MLflow tracking URI nếu có
        if self.config.mlflow_uri:
            mlflow.set_tracking_uri(self.config.mlflow_uri)
            logger.info(f"Connected to MLflow Tracking URI: {self.config.mlflow_uri}")

        with mlflow.start_run(run_name=f"{model_name}_Training"):
            # Huấn luyện mô hình trên tập train
            logger.info(f"Huấn luyện mô hình {model_name}...")
            model.fit(X_train, y_train)

            # Dự đoán trên tập validation
            y_pred = model.predict(X_val)
            y_prob = model.predict_proba(X_val)[:, 1]

            # Tính toán các metrics
            acc = accuracy_score(y_val, y_pred)
            prec = precision_score(y_val, y_pred, zero_division=0)
            rec = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_val, y_prob)

            metrics = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "roc_auc": roc_auc,
            }

            logger.info(f"Metrics cho {model_name}:")
            logger.info(f"  - Accuracy: {acc:.4f}")
            logger.info(f"  - Precision: {prec:.4f}")
            logger.info(f"  - Recall: {rec:.4f}")
            logger.info(f"  - F1-Score: {f1:.4f}")
            logger.info(f"  - ROC AUC: {roc_auc:.4f}")

            # Log parameters và metrics vào MLflow
            mlflow.log_params(params)
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            # Vẽ và log Confusion Matrix
            cm = confusion_matrix(y_val, y_pred)
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title(f"Confusion Matrix - {model_name}")
            plt.ylabel("Actual")
            plt.xlabel("Predicted")
            
            docs_dir = os.path.join("docs", "evaluation")
            os.makedirs(docs_dir, exist_ok=True)
            
            cm_path = os.path.join(docs_dir, f"{model_name}_cm.png")
            plt.savefig(cm_path, bbox_inches='tight')
            plt.close()
            mlflow.log_artifact(cm_path)

            # Vẽ và log ROC Curve
            fpr, tpr, _ = roc_curve(y_val, y_prob)
            roc_auc_val = auc(fpr, tpr)
            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc_val:.4f}")
            plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve - {model_name}")
            plt.legend(loc="lower right")
            
            roc_path = os.path.join(docs_dir, f"{model_name}_roc.png")
            plt.savefig(roc_path, bbox_inches='tight')
            plt.close()
            mlflow.log_artifact(roc_path)

            # Log model vào MLflow
            mlflow.sklearn.log_model(model, "model")

            logger.info(f"Đã log mô hình {model_name} vào MLflow")

            return {
                "model_name": model_name,
                "model_obj": model,
                "metrics": metrics,
            }

    def compare_models(self, results):
        """
        So sánh các mô hình và chọn mô hình tốt nhất dựa trên ROC AUC.
        
        Args:
            results: Danh sách kết quả từ các mô hình
            
        Returns:
            dict: Thông tin mô hình tốt nhất
        """
        best_model_info = max(results, key=lambda x: x["metrics"]["roc_auc"])
        logger.info(
            f"Mô hình tốt nhất: {best_model_info['model_name']} "
            f"với ROC AUC: {best_model_info['metrics']['roc_auc']:.4f}"
        )
        return best_model_info

    def save_model(self, best_model_info):
        """
        Lưu mô hình tốt nhất và metrics vào disk.
        
        Args:
            best_model_info: Thông tin mô hình tốt nhất
        """
        model_path = os.path.join(self.config.root_dir, self.config.model_name)
        joblib.dump(best_model_info["model_obj"], model_path)
        logger.info(f"Đã lưu mô hình tốt nhất tại: {model_path}")

        metrics_path = os.path.join(self.config.root_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(best_model_info["metrics"], f, indent=4)
        logger.info(f"Đã lưu metrics tại: {metrics_path}")

    def initiate_model_trainer(self):
        """
        Điểm vào chính của Model Trainer.
        Thực hiện toàn bộ quy trình: load data, tune, train, compare, save.
        """
        logger.info("Bắt đầu Stage 04: Model Training")

        # Bước 1: Load dữ liệu đã transform
        X_train_full, y_train_full, X_test, y_test = self.load_transformed_data()

        # Bước 2: Chia tập train thành train và validation (80-20)
        logger.info("Chia tập train thành train (80%) và validation (20%)")
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
        )
        logger.info(f"X_train: {X_train.shape}, X_val: {X_val.shape}")

        results = []

        # Bước 3: Huấn luyện LightGBM
        logger.info("=" * 60)
        logger.info("Tuning và Training LightGBM...")
        logger.info("=" * 60)
        
        best_lgbm, best_lgbm_params = self.hyperparameter_tuning(
            "LightGBM", LGBMClassifier, self.config.lgbm_params, X_train, y_train
        )
        lgbm_res = self.train_and_log(
            "LightGBM", best_lgbm, best_lgbm_params, X_train, X_val, y_train, y_val
        )
        results.append(lgbm_res)

        # Bước 4: Huấn luyện XGBoost
        logger.info("=" * 60)
        logger.info("Tuning và Training XGBoost...")
        logger.info("=" * 60)
        
        best_xgb, best_xgb_params = self.hyperparameter_tuning(
            "XGBoost", XGBClassifier, self.config.xgboost_params, X_train, y_train
        )
        xgb_res = self.train_and_log(
            "XGBoost", best_xgb, best_xgb_params, X_train, X_val, y_train, y_val
        )
        results.append(xgb_res)

        # Bước 5: Huấn luyện CatBoost
        logger.info("=" * 60)
        logger.info("Tuning và Training CatBoost...")
        logger.info("=" * 60)
        
        best_cb, best_cb_params = self.hyperparameter_tuning(
            "CatBoost", CatBoostClassifier, self.config.catboost_params, X_train, y_train
        )
        cb_res = self.train_and_log(
            "CatBoost", best_cb, best_cb_params, X_train, X_val, y_train, y_val
        )
        results.append(cb_res)

        # Bước 6: Huấn luyện StackingClassifier
        logger.info("=" * 60)
        logger.info("Huấn luyện mô hình Stacking (Ensemble)...")
        logger.info("=" * 60)
        
        stacking_model = StackingClassifier(
            estimators=[
                ('lgbm', best_lgbm),
                ('xgb', best_xgb),
                ('cb', best_cb)
            ],
            final_estimator=LogisticRegression(),
            cv=5
        )
        
        stacking_res = self.train_and_log(
            "StackingClassifier", stacking_model, {"estimators": "LGBM, XGB, CatBoost"}, X_train, X_val, y_train, y_val
        )
        results.append(stacking_res)

        # Bước 7: So sánh và chọn mô hình tốt nhất
        logger.info("=" * 60)
        logger.info("So sánh các mô hình...")
        logger.info("=" * 60)
        best_model_info = self.compare_models(results)

        # Bước 8: Lưu mô hình tốt nhất
        self.save_model(best_model_info)

        logger.info("Hoàn thành Stage 04: Model Training")
