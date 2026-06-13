import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

from src.entity.config_entity import DataTransformationConfig
from src.utils.common import logger


# =============================================================================
# Transformer tạo 10 đặc trưng phái sinh từ kết quả EDA
# Stateless: fit() không làm gì, chỉ transform() tạo các cột mới
# Phải đặt TRƯỚC ColumnTransformer để các cột mới được xử lý tiếp theo
# =============================================================================
class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()

        # ------------------------------------------------------------------
        # 1. loyalty_tier — Phân khúc trung thành
        #    Bins: [0, 6, 12, 24, 48, +inf]
        #    Labels: 0=Onboarding, 1=First Year, 2=Second Year,
        #            3=Familiar, 4=Loyal
        # ------------------------------------------------------------------
        # Dùng astype(float) thay vì astype(int)
        # Nếu tenure có NaN, pd.cut trả về NaN → astype(int) crash với ValueError.
        # astype(float) giữ nguyên NaN để SimpleImputer(median) ở bước sau xử lý đúng.
        X["loyalty_tier"] = pd.cut(
            X["tenure"],
            bins=[0, 6, 12, 24, 48, np.inf],
            labels=[0, 1, 2, 3, 4],
            right=True,
            include_lowest=True,
        ).astype(float)

        # ------------------------------------------------------------------
        # 2. charge_segment — Phân khúc cước phí hàng tháng
        #    Bins: [0, 35, 70, +inf]
        #    Labels: 0=Budget, 1=Standard, 2=Premium
        # ------------------------------------------------------------------
        # Dùng astype(float) thay vì astype(int)
        # Nếu tenure có NaN, pd.cut trả về NaN → astype(int) crash với ValueError.
        # astype(float) giữ nguyên NaN để SimpleImputer(median) ở bước sau xử lý đúng.
        X["charge_segment"] = pd.cut(
            X["MonthlyCharges"],
            bins=[0, 35, 70, np.inf],
            labels=[0, 1, 2],
            right=True,
            include_lowest=True,
        ).astype(float)

        # ------------------------------------------------------------------
        # 3. total_active_services — Tổng số dịch vụ đang sử dụng
        #    Đếm "Yes" của 7 dịch vụ tùy chọn +
        #    InternetService (DSL hoặc Fiber optic = 1) +
        #    PhoneService (Yes = 1) +
        #    MultipleLines (Yes = 1)
        # ------------------------------------------------------------------
        X["total_active_services"] = (
            (X["InternetService"].isin(["DSL", "Fiber optic"])).astype(int)
            + (X["OnlineSecurity"]   == "Yes").astype(int)
            + (X["TechSupport"]      == "Yes").astype(int)
            + (X["OnlineBackup"]     == "Yes").astype(int)
            + (X["DeviceProtection"] == "Yes").astype(int)
            + (X["StreamingTV"]      == "Yes").astype(int)
            + (X["StreamingMovies"]  == "Yes").astype(int)
            + (X["PhoneService"]     == "Yes").astype(int)
            + (X["MultipleLines"]    == "Yes").astype(int)
        )

        # ------------------------------------------------------------------
        # 4. charge_to_tenure_ratio_log — Log áp lực chi phí
        #    Công thức: np.log1p(MonthlyCharges / tenure)
        #    tenure=0 → tránh chia 0 bằng replace(0, NaN) rồi fillna(0)
        # ------------------------------------------------------------------
        # Tách biệt hai loại NaN:
        #   (a) tenure=0 (thiết kế): tránh chia-cho-0, kết quả log1p(NaN) cần gán về 0.
        #   (b) MonthlyCharges thực sự khuyết (NaN thật): GIỮ NGUYÊN NaN để
        #       SimpleImputer(strategy="median") ở bước sau điền đúng giá trị.
        # Cách làm: tính tỷ lệ, sau đó chỉ fillna(0) ở những dòng mà tenure=0.
        _tenure_safe = X["tenure"].replace(0, np.nan)  # tenure=0 → NaN
        _ratio_log   = np.log1p(X["MonthlyCharges"] / _tenure_safe)
        # Chỉ gán 0 khi NaN bắt nguồn từ tenure=0; NaN thật (MonthlyCharges thiếu) vẫn là NaN.
        X["charge_to_tenure_ratio_log"] = _ratio_log.where(
            ~(X["tenure"] == 0), other=0
        )

        # ------------------------------------------------------------------
        # 5. average_cost_per_service — Đơn giá trung bình mỗi dịch vụ
        #    Công thức: MonthlyCharges / total_active_services
        #    total_active_services=0 → fillna(0)
        # ------------------------------------------------------------------
        X["average_cost_per_service"] = (
            X["MonthlyCharges"] / X["total_active_services"].replace(0, np.nan)
        ).fillna(0)

        # ------------------------------------------------------------------
        # 6. security_score — Điểm Khiên Bảo vệ (thang [-1, 0..4])
        #    Đếm "Yes" trong 4 dịch vụ bảo vệ/hỗ trợ
        #    InternetService = "No" → gán đè thành -1
        # ------------------------------------------------------------------
        protective_cols = [
            "OnlineSecurity", "TechSupport",
            "OnlineBackup", "DeviceProtection",
        ]
        X["security_score"] = (X[protective_cols] == "Yes").sum(axis=1)
        X.loc[X["InternetService"] == "No", "security_score"] = -1

        # ------------------------------------------------------------------
        # 7. streaming_score — Điểm Giải trí (thang [-1, 0..2])
        #    Đếm "Yes" trong StreamingTV và StreamingMovies
        #    InternetService = "No" → gán đè thành -1
        # ------------------------------------------------------------------
        X["streaming_score"] = (
            X[["StreamingTV", "StreamingMovies"]] == "Yes"
        ).sum(axis=1)
        X.loc[X["InternetService"] == "No", "streaming_score"] = -1

        # ------------------------------------------------------------------
        # 8. manual_payment — Cờ thanh toán thủ công (0/1)
        #    1 nếu PaymentMethod là "Electronic check" hoặc "Mailed check"
        # ------------------------------------------------------------------
        X["manual_payment"] = X["PaymentMethod"].isin(
            ["Electronic check", "Mailed check"]
        ).astype(int)

        # ------------------------------------------------------------------
        # 9. composite_risk_profile — Siêu cờ tổ hợp rủi ro đỉnh điểm (0/1)
        #    1 nếu Contract="Month-to-month" ĐỒNG THỜI InternetService="Fiber optic"
        # ------------------------------------------------------------------
        X["composite_risk_profile"] = (
            (X["Contract"] == "Month-to-month") &
            (X["InternetService"] == "Fiber optic")
        ).astype(int)

        # ------------------------------------------------------------------
        # 10. demographic_profile — Phân khúc nhân khẩu học (0–3)
        #     0 = Single Youth    : SeniorCitizen=No  AND Partner=No  AND Dependents=No
        #     1 = Nuclear Family  : SeniorCitizen=No  AND (Partner=Yes OR Dependents=Yes)
        #     2 = Isolated Senior : SeniorCitizen=Yes AND Partner=No  AND Dependents=No
        #     3 = Supported Senior: SeniorCitizen=Yes AND (Partner=Yes OR Dependents=Yes)
        # ------------------------------------------------------------------
        is_senior   = X["SeniorCitizen"] == 1
        has_support = (X["Partner"] == "Yes") | (X["Dependents"] == "Yes")

        X["demographic_profile"] = np.select(
            condlist=[
                ~is_senior & ~has_support,   # 0: Single Youth
                ~is_senior &  has_support,   # 1: Nuclear Family
                 is_senior & ~has_support,   # 2: Isolated Senior
                 is_senior &  has_support,   # 3: Supported Senior
            ],
            choicelist=[0, 1, 2, 3],
            default=0,
        )

        return X


# =============================================================================
# Transformer cắt bớt outlier bằng phương pháp Winsorization
# fit() học ngưỡng cắt từ tập train, transform() áp dụng cho cả train và test
# =============================================================================
class WinsorizerTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, lower_pct=0.01, upper_pct=0.99):
        # Ngưỡng dưới (percentile 1%) và ngưỡng trên (percentile 99%)
        self.lower_pct = lower_pct
        self.upper_pct = upper_pct
        self.bounds_ = {}

    def fit(self, X, y=None):
        # Học ngưỡng cắt cho từng cột từ tập train
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        for i, col in enumerate(X.columns):
            self.bounds_[i] = (
                X[col].quantile(self.lower_pct),
                X[col].quantile(self.upper_pct),
            )
        return self

    def transform(self, X, y=None):
        # Cắt giá trị nằm ngoài ngưỡng đã học
        X = np.array(X, dtype=float).copy()
        for i, (lower, upper) in self.bounds_.items():
            X[:, i] = np.clip(X[:, i], lower, upper)
        return X


# =============================================================================
# Class chính của Stage 03, được gọi bởi src/pipeline/stage_03_data_transformation.py
# =============================================================================
class DataTransformation:

    # -----------------------------------------------------------------------
    # Các cột bị loại bỏ trước khi đưa vào pipeline
    # -----------------------------------------------------------------------
    COLS_TO_DROP = [
        "id",                      # Mã định danh, không có giá trị dự báo
        "gender",                  # Zero-Signal: spread churn không đáng kể
        "TotalCharges",            # Đa cộng tuyến nghiêm trọng với tenure
        "tenure",                  # Đã thay thế bởi loyalty_tier
        "MonthlyCharges",          # Đã thay thế bởi charge_segment
        "SeniorCitizen",           # Đã nén vào demographic_profile
        "Partner",                 # Đã nén vào demographic_profile
        "Dependents",              # Đã nén vào demographic_profile
        "bill_shock_ratio",        # Zero-Signal: không có khả năng phân tách
        "zero_supportive_service", # Đa cộng tuyến hoàn hảo (V=1.00) với security_score
    ]

    # -----------------------------------------------------------------------
    # Cột số: SimpleImputer(median) → WinsorizerTransformer → StandardScaler
    # Gồm 5 feature phái sinh dạng liên tục/thứ bậc số
    # -----------------------------------------------------------------------
    NUMERIC_COLS = [
        "charge_to_tenure_ratio_log",  # Log áp lực chi phí
        "average_cost_per_service",    # Đơn giá trung bình/dịch vụ
        "total_active_services",       # Tổng số dịch vụ đang dùng (0–9)
        "security_score",              # Điểm Khiên Bảo vệ (-1 đến 4)
        "streaming_score",             # Điểm Giải trí (-1 đến 2)
    ]

    # -----------------------------------------------------------------------
    # Cột phân loại gốc nhiều nhãn: SimpleImputer → OneHotEncoder(drop='first')
    # -----------------------------------------------------------------------
    CATEGORICAL_COLS = [
        "MultipleLines",     # Yes / No / No phone service
        "InternetService",   # DSL / Fiber optic / No
        "OnlineSecurity",    # Yes / No / No internet service
        "OnlineBackup",      # Yes / No / No internet service
        "DeviceProtection",  # Yes / No / No internet service
        "TechSupport",       # Yes / No / No internet service
        "StreamingTV",       # Yes / No / No internet service
        "StreamingMovies",   # Yes / No / No internet service
        "Contract",          # Month-to-month / One year / Two year
        "PaymentMethod",     # 4 phương thức thanh toán
    ]

    # -----------------------------------------------------------------------
    # Cột nhị phân Yes/No gốc: SimpleImputer → OneHotEncoder(drop='first')
    # -----------------------------------------------------------------------
    BINARY_COLS = [
        "PhoneService",      # Yes / No
        "PaperlessBilling",  # Yes / No
    ]

    # -----------------------------------------------------------------------
    # Cột thứ bậc phái sinh (ordinal): chỉ SimpleImputer
    # -----------------------------------------------------------------------
    ORDINAL_COLS = [
        "loyalty_tier",        # 0–4 thứ bậc trung thành
        "charge_segment",      # 0–2 phân khúc cước phí
        "demographic_profile", # 0–3 phân khúc nhân khẩu học
    ]

    # -----------------------------------------------------------------------
    # Cột nhị phân phái sinh (0/1): chỉ SimpleImputer
    # -----------------------------------------------------------------------
    ENGINEERED_BINARY_COLS = [
        "manual_payment",         # 0/1
        "composite_risk_profile", # 0/1
    ]

    TARGET_COL = "Churn"

    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def get_data_transformer_object(self) -> Pipeline:
        # Xây dựng sklearn Pipeline gồm 2 bước chính:
        # Bước 1: ChurnFeatureEngineer tạo feature mới trước
        # Bước 2: ColumnTransformer xử lý từng nhóm cột khác nhau
        # Trả về pipeline CHƯA fit

        logger.info("Xây dựng preprocessing pipeline")

        # Pipeline cho cột số: xử lý giá trị thiếu bằng median, cắt outlier, chuẩn hóa
        numeric_pipeline = Pipeline(steps=[
            ("imputer",    SimpleImputer(strategy="median")),
            ("winsorizer", WinsorizerTransformer()),
            ("scaler",     StandardScaler()),
        ])

        # Pipeline cho cột phân loại: xử lý giá trị thiếu bằng giá trị phổ biến nhất, mã hóa OHE
        categorical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(
                handle_unknown="ignore",  # bỏ qua category lạ trong tập test
                drop="first",             # tránh dummy variable trap
                sparse_output=False,
            )),
        ])

        # Pipeline cho cột nhị phân Yes/No: xử lý giá trị thiếu, mã hóa OHE thành 0/1
        binary_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(
                handle_unknown="ignore",
                drop="first",
                sparse_output=False,
            )),
        ])

        # Pipeline cho cột thứ bậc phái sinh: chỉ cần xử lý giá trị thiếu
        ordinal_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ])

        # Pipeline cho các feature nhị phân đã tạo sẵn: chỉ cần xử lý giá trị thiếu
        engineered_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ])

        # Gộp tất cả các pipeline lại theo từng nhóm cột
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline,     self.NUMERIC_COLS),
                ("cat", categorical_pipeline, self.CATEGORICAL_COLS),
                ("bin", binary_pipeline,      self.BINARY_COLS),
                ("ord", ordinal_pipeline,     self.ORDINAL_COLS),
                ("eng", engineered_pipeline,  self.ENGINEERED_BINARY_COLS),
            ],
            remainder="drop",  # loại bỏ tất cả cột không được liệt kê ở trên
        )

        # Pipeline đầy đủ: tạo feature mới trước, rồi xử lý
        full_pipeline = Pipeline(steps=[
            ("feature_engineering", ChurnFeatureEngineer()),
            ("preprocessing",       preprocessor),
        ])

        return full_pipeline

    def initiate_data_transformation(self):
        # Điểm vào được gọi bởi stage_03_data_transformation.py
        # Thứ tự xử lý: load dữ liệu, lọc hàng, drop cột, tách X/y,
        # fit pipeline trên train, transform test, SMOTE trên train, lưu kết quả

        logger.info("Bắt đầu Stage 03: Data Transformation")

        # Bước 1: Đọc dữ liệu từ artifacts của Stage 01
        logger.info(f"Đọc file train: {self.config.train_data_path}")
        df_train = pd.read_csv(self.config.train_data_path)

        logger.info(f"Đọc file test: {self.config.test_data_path}")
        df_test = pd.read_csv(self.config.test_data_path)

        logger.info(f"Kích thước train: {df_train.shape}, kích thước test: {df_test.shape}")

        # Bước 2: Chuyển TotalCharges sang số (trong dataset có thể bị lưu dạng chuỗi do khoảng trắng)
        for df in [df_train, df_test]:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

        # Bước 3: Lọc bỏ các hàng có InternetService="No" VÀ PhoneService="No"
        # Nhóm này không sử dụng bất kỳ dịch vụ nào, không có giá trị dự báo
        before = len(df_train)
        df_train = df_train[
            ~((df_train["InternetService"] == "No") & (df_train["PhoneService"] == "No"))
        ].reset_index(drop=True)
        logger.info(f"Lọc Internet=No & Phone=No: {before} → {len(df_train)} hàng (bỏ {before - len(df_train)})")

        if "InternetService" in df_test.columns and "PhoneService" in df_test.columns:
            before_test = len(df_test)
            df_test = df_test[
                ~((df_test["InternetService"] == "No") & (df_test["PhoneService"] == "No"))
            ].reset_index(drop=True)
            logger.info(f"Lọc test Internet=No & Phone=No: {before_test} → {len(df_test)} hàng")

        # Bước 4: (Bỏ qua) Không drop cột ở đây vì ChurnFeatureEngineer cần chúng để tính toán.
        # Các cột gốc này sẽ tự động bị loại bỏ bởi ColumnTransformer(remainder="drop") ở cuối pipeline.

        # Bước 5: Tách biến đầu vào X và nhãn mục tiêu y
        X_train = df_train.drop(columns=[self.TARGET_COL])
        y_train = df_train[self.TARGET_COL].map({"Yes": 1, "No": 0})

        # Tập test của Kaggle có thể không có cột target
        if self.TARGET_COL in df_test.columns:
            X_test = df_test.drop(columns=[self.TARGET_COL])
            y_test = df_test[self.TARGET_COL].map({"Yes": 1, "No": 0})
        else:
            X_test = df_test.copy()
            y_test = None

        logger.info(f"X_train: {X_train.shape}, tỷ lệ churn: {y_train.mean():.2%}")

        # Bước 6: Fit pipeline CHỈ trên tập train để tránh data leakage
        logger.info("Fit preprocessing pipeline trên X_train")
        pipeline = self.get_data_transformer_object()
        X_train_transformed = pipeline.fit_transform(X_train)

        # Bước 7: Chỉ transform X_test, không fit lại
        logger.info("Transform X_test (không fit lại)")
        X_test_transformed = pipeline.transform(X_test)

        logger.info(f"Sau transform: X_train {X_train_transformed.shape}, X_test {X_test_transformed.shape}")

        # Bước 8: Áp dụng SMOTE CHỈ trên tập train để xử lý mất cân bằng nhãn
        logger.info("Áp dụng SMOTE trên X_train để cân bằng nhãn")
        logger.info(f"Phân phối nhãn trước SMOTE: {dict(zip(*np.unique(y_train, return_counts=True)))}")

        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(
            X_train_transformed, y_train
        )

        logger.info(f"Phân phối nhãn sau SMOTE: {dict(zip(*np.unique(y_train_resampled, return_counts=True)))}")
        logger.info(f"X_train sau SMOTE: {X_train_resampled.shape}")

        # Bước 9: Lưu pipeline đã fit (không lưu SMOTE vì không cần khi dự báo)
        joblib.dump(pipeline, self.config.preprocessor_path)
        logger.info(f"Đã lưu preprocessor: {self.config.preprocessor_path}")

        # Bước 10: Lưu dữ liệu đã xử lý để Stage 04 sử dụng
        train_path = self.config.root_dir / "train_transformed.npz"
        test_path  = self.config.root_dir / "test_transformed.npz"

        np.savez(train_path, X=X_train_resampled, y=np.array(y_train_resampled))

        if y_test is not None:
            np.savez(test_path, X=X_test_transformed, y=np.array(y_test))
        else:
            np.savez(test_path, X=X_test_transformed)

        logger.info(f"Đã lưu file train: {train_path}")
        logger.info(f"Đã lưu file test: {test_path}")

        # Bước 11: Kiểm tra kết quả trước khi kết thúc
        assert X_train_resampled.shape[1] == X_test_transformed.shape[1], \
            "Lỗi: Số cột của train và test không khớp sau khi transform"
        assert not np.isnan(X_train_resampled).any(), \
            "Lỗi: Còn giá trị NaN trong X_train sau khi transform"
        assert not np.isnan(X_test_transformed).any(), \
            "Lỗi: Còn giá trị NaN trong X_test sau khi transform"

        logger.info(f"Kiểm tra thành công, số feature cuối: {X_train_resampled.shape[1]}")
        logger.info("Hoàn thành Stage 03: Data Transformation")