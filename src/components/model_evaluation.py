import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
import mlflow
import mlflow.sklearn
from src.utils.logger import logger
from src.entity.config_entity import ModelEvaluationConfig
from urllib.parse import urlparse


class ModelEvaluation:
    """
    Component chịu trách nhiệm đánh giá mô hình (Model Evaluation).
    Tính toán các metrics, tạo biểu đồ và log kết quả vào MLflow.
    """
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def load_test_data(self):
        """
        Đọc dữ liệu test đã được transform từ file .npz.
        
        Returns:
            tuple: (X_test, y_test) hoặc (X_test, None) nếu không có nhãn
        """
        logger.info(f"Đọc dữ liệu test từ: {self.config.test_data_path}")
        test_data = np.load(self.config.test_data_path)
        X_test = test_data['X']
        
        # Kiểm tra xem có nhãn hay không
        if 'y' in test_data:
            y_test = test_data['y']
            logger.info(f"Kích thước X_test: {X_test.shape}, y_test: {y_test.shape}")
        else:
            y_test = None
            logger.warning("Tập test không có nhãn, chỉ có thể tạo predictions")
            logger.info(f"Kích thước X_test: {X_test.shape}")
        
        return X_test, y_test

    def save_results(self, metrics, cm, fpr, tpr, roc_auc):
        """
        Lưu metrics và các biểu đồ đánh giá.
        
        Args:
            metrics: Dictionary chứa các metrics
            cm: Confusion matrix
            fpr, tpr: False positive rate và True positive rate cho ROC curve
            roc_auc: Area under ROC curve
            
        Returns:
            tuple: (cm_path, roc_path) - Đường dẫn đến các file ảnh
        """
        # Lưu metrics.json
        with open(self.config.metric_file_name, "w") as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Đã lưu metrics tại: {self.config.metric_file_name}")

        # Tạo và lưu Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True)
        plt.title("Confusion Matrix", fontsize=16, fontweight='bold')
        plt.ylabel("Actual Label", fontsize=12)
        plt.xlabel("Predicted Label", fontsize=12)
        cm_path = os.path.join(self.config.root_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Đã lưu Confusion Matrix tại: {cm_path}")

        # Tạo và lưu ROC Curve
        plt.figure(figsize=(8, 6))
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label=f"ROC curve (AUC = {roc_auc:.4f})",
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        roc_path = os.path.join(self.config.root_dir, "roc_curve.png")
        plt.savefig(roc_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Đã lưu ROC Curve tại: {roc_path}")

        return cm_path, roc_path

    def initiate_model_evaluation(self):
        """
        Điểm vào chính của Model Evaluation.
        Thực hiện toàn bộ quy trình: load model, load data, predict, evaluate, visualize, log.
        """
        try:
            logger.info("Bắt đầu Stage 05: Model Evaluation")

            # Bước 1: Load mô hình đã được train
            logger.info(f"Đọc mô hình từ: {self.config.model_path}")
            model = joblib.load(self.config.model_path)
            logger.info(f"Đã load mô hình: {type(model).__name__}")

            # Bước 2: Load dữ liệu test
            X_test, y_test = self.load_test_data()

            # Kiểm tra xem có nhãn để đánh giá hay không
            if y_test is None:
                logger.warning("Không có nhãn test, không thể tính metrics. Chỉ tạo predictions.")
                # Tạo predictions và lưu
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]
                
                predictions_path = os.path.join(self.config.root_dir, "predictions.npz")
                np.savez(predictions_path, predictions=y_pred, probabilities=y_prob)
                logger.info(f"Đã lưu predictions tại: {predictions_path}")
                logger.info("Hoàn thành Stage 05: Model Evaluation (chỉ predictions)")
                return

            # Bước 3: Thiết lập MLflow
            if self.config.mlflow_uri:
                mlflow.set_tracking_uri(self.config.mlflow_uri)
                logger.info(f"Connected to MLflow Tracking URI: {self.config.mlflow_uri}")
            
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            with mlflow.start_run(run_name="Model_Evaluation"):
                # Bước 4: Dự đoán trên tập test
                logger.info("Thực hiện dự đoán trên tập test...")
                y_pred = model.predict(X_test)
                
                # Lấy xác suất dự đoán cho ROC curve
                try:
                    y_prob = model.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_prob)
                    roc_auc = auc(fpr, tpr)
                    roc_auc_score_val = roc_auc_score(y_test, y_prob)
                except Exception as e:
                    logger.warning(f"Không thể tính ROC AUC: {e}")
                    roc_auc = 0.0
                    roc_auc_score_val = 0.0
                    fpr, tpr = [0, 1], [0, 1]

                # Bước 5: Tính toán các metrics
                logger.info("Tính toán các metrics đánh giá...")
                metrics = {
                    "accuracy": float(accuracy_score(y_test, y_pred)),
                    "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                    "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                    "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
                    "roc_auc": float(roc_auc_score_val),
                }

                logger.info("Kết quả đánh giá mô hình:")
                logger.info(f"  - Accuracy:  {metrics['accuracy']:.4f}")
                logger.info(f"  - Precision: {metrics['precision']:.4f}")
                logger.info(f"  - Recall:    {metrics['recall']:.4f}")
                logger.info(f"  - F1-Score:  {metrics['f1_score']:.4f}")
                logger.info(f"  - ROC AUC:   {metrics['roc_auc']:.4f}")

                # Bước 6: Tạo confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                logger.info(f"Confusion Matrix:\n{cm}")

                # Bước 7: Log metrics vào MLflow
                logger.info("Logging metrics vào MLflow...")
                for key, value in metrics.items():
                    mlflow.log_metric(key, value)

                # Bước 8: Lưu và log artifacts (biểu đồ)
                logger.info("Tạo và lưu các biểu đồ đánh giá...")
                cm_path, roc_path = self.save_results(metrics, cm, fpr, tpr, roc_auc)

                mlflow.log_artifact(cm_path)
                mlflow.log_artifact(roc_path)
                mlflow.log_artifact(str(self.config.metric_file_name))

                # Bước 9: Log parameters từ training
                logger.info("Logging parameters vào MLflow...")
                # Convert ConfigBox to dict if needed
                params_dict = dict(self.config.all_params) if hasattr(self.config.all_params, '__dict__') else self.config.all_params
                mlflow.log_params(params_dict)

                # Bước 10: Log model vào MLflow
                if tracking_url_type_store != "file":
                    # Đăng ký model nếu không phải local file store
                    mlflow.sklearn.log_model(
                        model, "model", registered_model_name="ChurnPredictionModel"
                    )
                    logger.info("Đã đăng ký model vào MLflow Model Registry")
                else:
                    mlflow.sklearn.log_model(model, "model")
                    logger.info("Đã log model vào MLflow (local)")

            logger.info("MLflow Evaluation run hoàn thành")
            logger.info("Hoàn thành Stage 05: Model Evaluation")

        except Exception as e:
            logger.error(f"Lỗi trong quá trình Model Evaluation: {e}")
            raise e
