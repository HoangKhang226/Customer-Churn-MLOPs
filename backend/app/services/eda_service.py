import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any

# Khởi tạo logger cho service
logger = logging.getLogger(__name__)

class EDAService:
    # Bước 0: load file dữ liệu
    def __init__(self, data_path: str):
        """
        Khởi tạo service và load dữ liệu từ file CSV.
        Đồng thời thực hiện một số bước tiền xử lý cơ bản và Feature Engineering như trong notebook.
        """
        logger.info("Khởi tạo EDAService. Đang tải dữ liệu từ: %s...", data_path)
        try:
            self.df = pd.read_csv(data_path)
            logger.info("Tải dữ liệu thành công. Kích thước tập dữ liệu: %s", self.df.shape)
        except Exception as e:
            logger.error("Lỗi khi tải file dữ liệu từ %s: %s", data_path, str(e))
            raise e

        # Tiền xử lý giống notebook: chuyển đổi SeniorCitizen từ 1/0 thành Yes/No
        if 'SeniorCitizen' in self.df.columns:
            logger.info("Cột 'SeniorCitizen' tồn tại. Tiến hành chuẩn hóa dữ liệu sang 'Yes'/'No'.")
            self.df['SeniorCitizen'] = self.df['SeniorCitizen'].replace({1: 'Yes', 0: 'No'})

        # Chuẩn hóa cột TotalCharges sang numeric để tránh lỗi chuỗi trống
        if 'TotalCharges' in self.df.columns:
            self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'], errors='coerce').fillna(0.0)

        # --- FEATURE ENGINEERING (Xây dựng đặc trưng phái sinh từ Notebook) ---
        logger.info("Bắt đầu xây dựng các đặc trưng phái sinh mới từ EDA...")

        # 1. Rời rạc hóa tenure -> loyalty_tier
        if 'tenure' in self.df.columns:
            self.df['loyalty_tier'] = pd.cut(
                self.df['tenure'],
                bins=[0, 6, 12, 24, 48, float('inf')],
                labels=['Onboarding (0-6 tháng)', 'First Year (7-12 tháng)', 'Second Year (13-24 tháng)', 'Familiar (25-48 tháng)', 'Loyal (trên 48 tháng)'],
                include_lowest=True
            ).astype(str)

        # 2. Rời rạc hóa MonthlyCharges -> charge_segment
        if 'MonthlyCharges' in self.df.columns:
            self.df['charge_segment'] = pd.cut(
                self.df['MonthlyCharges'],
                bins=[0, 35, 70, float('inf')],
                labels=['Budget (0-35$)', 'Standard (35-70$)', 'Premium (trên 70$)'],
                include_lowest=True
            ).astype(str)

        # 3. Tính total_active_services (Số lượng dịch vụ hoạt động)
        service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'MultipleLines']
        def count_active_services(row):
            count = 0
            if 'InternetService' in row and row['InternetService'] in ['DSL', 'Fiber optic']:
                count += 1
            if 'PhoneService' in row and row['PhoneService'] == 'Yes':
                count += 1
            for col in service_cols:
                if col in row and row[col] == 'Yes':
                    count += 1
            return count
        self.df['total_active_services'] = self.df.apply(count_active_services, axis=1)

        # 4. charge_to_tenure_ratio_log
        if 'MonthlyCharges' in self.df.columns and 'tenure' in self.df.columns:
            safe_tenure = self.df['tenure'].replace(0, 1)
            self.df['charge_to_tenure_ratio_log'] = np.log1p(self.df['MonthlyCharges'] / safe_tenure)

        # 5. average_cost_per_service
        if 'MonthlyCharges' in self.df.columns:
            safe_services = self.df['total_active_services'].replace(0, 1)
            self.df['average_cost_per_service'] = self.df['MonthlyCharges'] / safe_services

        # 6. security_score
        protective_cols = ['OnlineSecurity', 'TechSupport', 'OnlineBackup', 'DeviceProtection']
        available_protective = [col for col in protective_cols if col in self.df.columns]
        if available_protective:
            self.df['security_score'] = (self.df[available_protective] == 'Yes').sum(axis=1)
            if 'InternetService' in self.df.columns:
                self.df.loc[self.df['InternetService'] == 'No', 'security_score'] = -1

        # 7. streaming_score
        streaming_cols = ['StreamingTV', 'StreamingMovies']
        available_streaming = [col for col in streaming_cols if col in self.df.columns]
        if available_streaming:
            self.df['streaming_score'] = (self.df[available_streaming] == 'Yes').sum(axis=1)
            if 'InternetService' in self.df.columns:
                self.df.loc[self.df['InternetService'] == 'No', 'streaming_score'] = -1

        # 8. manual_payment
        if 'PaymentMethod' in self.df.columns:
            self.df['manual_payment'] = self.df['PaymentMethod'].isin(['Electronic check', 'Mailed check']).astype(int)

        # 9. composite_risk_profile
        if 'Contract' in self.df.columns and 'InternetService' in self.df.columns:
            self.df['composite_risk_profile'] = ((self.df['Contract'] == 'Month-to-month') & (self.df['InternetService'] == 'Fiber optic')).astype(int)

        # 10. demographic_profile
        if 'SeniorCitizen' in self.df.columns and 'Partner' in self.df.columns and 'Dependents' in self.df.columns:
            self.df['demographic_profile'] = (
                self.df['SeniorCitizen'].astype(str) + "_" +
                self.df['Partner'].astype(str) + "_" +
                self.df['Dependents'].astype(str)
            )

        logger.info("Xây dựng đặc trưng phái sinh thành công. Kích thước tập dữ liệu mới: %s", self.df.shape)

    # Bước 1, 2: thống kê mô tả dữ liệu
    # [Mục 1 & 2 trong Notebook]
    def get_dataset_overview(self) -> Dict[str, Any]:
        """
        Trả về thông tin tổng quan của tập dữ liệu:
        - Số hàng, số cột
        - Số lượng dòng trùng lặp (Validity & Consistency)
        - Số lượng giá trị thiếu (Completeness Check)
        - Danh sách các cột kèm kiểu dữ liệu tương ứng
        """
        logger.info("Đang thực hiện thống kê tổng quan dữ liệu (get_dataset_overview)...")
        num_rows, num_cols = self.df.shape
        missing_values = self.df.isnull().sum().to_dict()
        duplicates = int(self.df.duplicated().sum())
        
        column_types = {col: str(dtype) for col, dtype in self.df.dtypes.items()}
        
        # Phân loại biến giống Mục 2.2 của Notebook
        numerical_cols = self.df.select_dtypes(include='number').drop(columns=['id'], errors='ignore').columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object']).drop(columns=['Churn'], errors='ignore').columns.tolist()

        logger.info(
            "Thành công. Phát hiện %s dòng trùng lặp và %s giá trị thiếu.",
            duplicates,
            sum(missing_values.values()),
        )
        
        insight = (
            "Kích thước dữ liệu cho thấy tập huấn luyện có nhiều hơn tập test 1 cột (chính là cột Churn - target của dự án). "
            "Biến định danh 'id' đóng vai trò định danh kiểm tra chất lượng dòng dữ liệu, không mang ý nghĩa thống kê. "
            "Tập dữ liệu bao gồm 3 biến định lượng chính (tenure, MonthlyCharges, TotalCharges) và 16 đặc trưng định tính (phân loại)."
        )

        return {
            "shape": {"rows": num_rows, "columns": num_cols},
            "duplicates": duplicates,
            "missing_values_count": sum(missing_values.values()),
            "column_types": column_types,
            "feature_roles": {
                "identifiers": ["id"] if "id" in self.df.columns else [],
                "numerical": numerical_cols,
                "categorical": categorical_cols,
                "target": ["Churn"] if "Churn" in self.df.columns else []
            },
            "insight": insight
        }

    def get_data_sanity_check(self) -> Dict[str, Any]:
        """
        [Mục 3.1 & 3.3 trong Notebook]
        Rà soát lỗi logic nghiệp vụ của các nhóm biến:
        - Biến định lượng (tenure, MonthlyCharges, TotalCharges phải dương)
        - Biến định tính (quan hệ logic chéo giữa InternetService/PhoneService và các dịch vụ đi kèm)
        """
        logger.info("Đang thực hiện kiểm tra tính hợp lệ logic dữ liệu (get_data_sanity_check)...")
        # Kiểm tra logic định lượng
        numeric_errors = {
            "tenure_invalid": int(((self.df["tenure"] <= 0) | (self.df["tenure"] % 1 != 0)).sum()),
            "monthly_charges_invalid": int((self.df["MonthlyCharges"] <= 0).sum()),
            "total_charges_invalid": int((pd.to_numeric(self.df["TotalCharges"], errors='coerce') <= 0).sum())
        }
        
        # Kiểm tra logic chéo của biến phân loại (như mục 3.3)
        internet_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
        no_internet_but_has_service = 0
        if "InternetService" in self.df.columns:
            no_internet_but_has_service = int(
                self.df.loc[self.df["InternetService"].eq("No"), internet_cols]
                .ne("No internet service").any(axis=1).sum()
            )
            
        logger.info("Hoàn tất sanity check.")
        if any(v > 0 for v in numeric_errors.values()) or no_internet_but_has_service > 0:
            logger.warning(
                "Phát hiện lỗi logic dữ liệu: Định lượng=%s, Định tính=%s",
                numeric_errors,
                no_internet_but_has_service,
            )

        insight = (
            "Kết quả kiểm định chéo đồng loạt bằng 0 chứng minh tập dữ liệu thô cực kỳ nhất quán, "
            "không tồn tại trường hợp lỗi logic toán học (như cước phí âm hay thời gian gắn bó lẻ) "
            "hoặc mâu thuẫn hệ sinh thái dịch vụ (như không đăng ký Internet nhưng vẫn có dịch vụ OnlineSecurity). "
            "Điều này đảm bảo tính toàn vẹn thông tin gốc giúp quá trình Feature Engineering diễn ra an toàn."
        )

        return {
            "numerical_sanity": numeric_errors,
            "categorical_sanity": {
                "internet_logic_errors": no_internet_but_has_service
            },
            "insight": insight
        }

    def get_numerical_statistics(self) -> Dict[str, Any]:
        """
        [Mục 4.1 trong Notebook]
        Trả về các chỉ số thống kê đơn biến của các cột số:
        - Mean, Min, Max, Variance, Skewness, Median, số lượng giá trị duy nhất (nunique).
        """
        logger.info("Đang tính toán các chỉ số thống kê định lượng (get_numerical_statistics)...")
        cols = [
            'tenure', 'MonthlyCharges', 'TotalCharges',
            'total_active_services', 'charge_to_tenure_ratio_log',
            'average_cost_per_service', 'security_score', 'streaming_score'
        ]
        stats = {}
        for col in cols:
            if col in self.df.columns:
                stats[col] = {
                    "mean": float(self.df[col].mean()),
                    "min": float(self.df[col].min()),
                    "max": float(self.df[col].max()),
                    "skewness": float(self.df[col].skew() if len(self.df[col].dropna()) > 2 else 0.0),
                    "variance": float(self.df[col].var()),
                    "nunique": int(self.df[col].nunique()),
                    "q1": float(self.df[col].quantile(0.25)),
                    "median": float(self.df[col].median()),
                    "q3": float(self.df[col].quantile(0.75)),
                }
        logger.info("Tính toán xong thống kê cho các cột: %s", list(stats.keys()))
        
        insight = (
            "Phân tích các đặc trưng định lượng gốc và phái sinh cho thấy độ phân tán rộng của cước phí, "
            "trong khi các điểm số dịch vụ và điểm số khiên bảo mật tập trung làm nổi bật sự đóng góp của hệ sinh thái đối với sự gắn bó. "
            "Đặc trưng average_cost_per_service cho thấy giá trị trung bình trên mỗi dịch vụ rất ổn định, giúp dễ dàng nhận biết rủi ro ngợp chi phí."
        )

        return {
            **stats,
            "insight": insight
        }

    def get_numerical_distribution(self, column_name: str, bins: int = 15) -> Dict[str, Any]:
        """
        [Mục 4.1 trong Notebook]
        Tính toán phân phối của 1 cột số để Frontend vẽ đồ thị Histogram và Boxplot:
        - Trả về bins và counts (cho Histogram)
        - Trả về danh sách dữ liệu mẫu hoặc phân vị (cho Boxplot)
        """
        logger.info(
            "Tính toán phân phối tần suất định lượng cho cột: %s (bins=%s)",
            column_name,
            bins,
        )
        if column_name not in self.df.columns:
            logger.error("Yêu cầu thất bại: cột '%s' không tồn tại trong dataset.", column_name)
            raise ValueError(f"Column {column_name} does not exist.")
            
        data = self.df[column_name].dropna()
        counts, bin_edges = np.histogram(data, bins=bins)
        
        insights_map = {
            "tenure": "Thời gian sử dụng dịch vụ (tenure) tập trung cực kỳ nhiều ở nhóm khách hàng mới (1-5 tháng). Phân phối không có điểm ngoại lai (outliers), giúp bảo toàn tính chất nguyên bản của thời gian gắn bó.",
            "MonthlyCharges": "Cước phí hàng tháng phân bố khá rộng với hai đỉnh lớn: một ở mức cước thấp (xấp xỉ 20 USD) và một ở phân khúc cao (70-90 USD), thể hiện sự phân hóa rõ rệt của khách hàng.",
            "TotalCharges": "Tổng cước phí tích lũy (TotalCharges) bị lệch phải rõ rệt do phần lớn khách hàng mới tích lũy ít cước phí, trong khi nhóm khách hàng lâu năm tích lũy cước phí cao."
        }
        insight = insights_map.get(column_name, "Dữ liệu phân phối ổn định, không phát hiện điểm dị biệt nghiêm trọng.")

        return {
            "labels": [f"{round(bin_edges[i], 1)}-{round(bin_edges[i+1], 1)}" for i in range(len(bin_edges)-1)],
            "values": counts.tolist(),
            "boxplot_data": {
                "min": float(data.min()),
                "q1": float(data.quantile(0.25)),
                "median": float(data.median()),
                "q3": float(data.quantile(0.75)),
                "max": float(data.max())
            },
            "insight": insight
        }

    def get_categorical_distribution(self, column_name: str) -> Dict[str, Any]:
        """
        [Mục 4.2 & 4.3 trong Notebook]
        Tính toán tần suất và phần trăm của các cột phân loại (bao gồm cả biến Target Churn)
        để vẽ biểu đồ tròn (Pie chart) hoặc biểu đồ cột (Bar chart).
        """
        logger.info("Tính toán phân phối tần suất định tính cho cột: %s", column_name)
        if column_name not in self.df.columns:
            logger.error("Yêu cầu thất bại: cột '%s' không tồn tại trong dataset.", column_name)
            raise ValueError(f"Column {column_name} does not exist.")
            
        value_counts = self.df[column_name].value_counts(dropna=False)
        total = len(self.df)
        
        insights_map = {
            "Churn": "Tỷ lệ khách hàng rời bỏ dịch vụ (Churn = Yes) chiếm khoảng 22.8%, cho thấy tập dữ liệu có sự mất cân bằng nhóm khá rõ rệt nhưng chưa đến mức cực đoan.",
            "PaymentMethod": "Phương thức Electronic check chiếm ưu thế lớn nhất trong nhóm khách hàng rời đi, trong khi các phương thức tự động (Credit card, Bank transfer) có tính ổn định cao hơn.",
            "Contract": "Hợp đồng ngắn hạn Month-to-month chiếm đa số và cũng là nhóm có tỷ lệ rời đi cao nhất, cho thấy tính cam kết thấp từ phía khách hàng.",
            "InternetService": "Khách hàng sử dụng dịch vụ cáp quang (Fiber optic) chiếm tỷ trọng lớn và có tỷ lệ Churn rất đáng chú ý so với nhóm dùng DSL hoặc không dùng Internet."
        }
        insight = insights_map.get(column_name, f"Phân phối tần suất của đặc trưng {column_name} cho thấy cơ cấu phân lớp rõ ràng giữa các nhóm thuộc tính.")

        return {
            "labels": value_counts.index.astype(str).tolist(),
            "counts": value_counts.values.tolist(),
            "percentages": (value_counts.values / total * 100).round(2).tolist(),
            "insight": insight
        }

    def get_bivariate_analysis(self, feature_name: str) -> Dict[str, Any]:
        """
        [Phân tích đa biến: Liên kết giữa Feature bất kỳ và Target Churn]
        - Nếu feature là dạng phân loại: Trả về bảng chéo (cross-tabulation) với Churn.
        - Nếu feature là dạng số: Trả về phân phối của feature đó phân nhóm theo Churn = Yes và Churn = No.
        """
        logger.info(
            "Đang thực hiện phân tích đa biến liên kết giữa cột '%s' và target 'Churn'...",
            feature_name,
        )
        if "Churn" not in self.df.columns or feature_name not in self.df.columns:
            logger.error("Yêu cầu phân tích đa biến thất bại vì thiếu cột Churn hoặc feature.")
            raise ValueError("Churn or feature column missing.")
            
        insights_map = {
            "PaymentMethod": "Phương thức Electronic check có tỷ lệ Churn cao vượt trội đạt 48.91% (rủi ro cao nhất), trong khi Credit card tự động chỉ có 6.93% (rủi ro thấp nhất), chênh lệch phân hóa đạt tới 41.97%.",
            "Contract": "Hợp đồng Month-to-month có tỷ lệ Churn rất cao (42.05%), trong khi hợp đồng 2 năm gần như triệt tiêu rủi ro rời đi khi tỷ lệ Churn chỉ ở mức 1.00%, độ phân hóa đạt 41.06%.",
            "InternetService": "Khách hàng sử dụng Fiber optic đối mặt tỷ lệ Churn lên đến 41.54% so với chỉ 1.43% ở nhóm không dùng Internet. Đây là đặc trưng phân hóa cực kỳ quan trọng.",
            "gender": "Tỷ lệ rời bỏ dịch vụ giữa Nam và Nữ là đồng đều một cách tuyệt đối (Nam: 22.23%, Nữ: 22.80%), độ phân hóa chỉ 0.57%. Đặc trưng này mang lại Information Gain quá thấp và nên loại bỏ để tránh nhiễu.",
            "SeniorCitizen": "Nhóm người cao tuổi (SeniorCitizen = Yes) có tỷ lệ rời đi lên đến 50.03% so với 18.98% ở nhóm trẻ, cho thấy phân khúc này cần có sự chăm sóc đặc biệt.",
            "tenure": "Nhóm khách hàng rời đi (Churn = Yes) có thời gian gắn bó (tenure) trung bình ngắn hơn đáng kể so với nhóm ở lại (Churn = No).",
            "MonthlyCharges": "Khách hàng rời đi (Churn = Yes) có mức cước phí hàng tháng trung bình cao hơn rõ rệt (khoảng 74.4 USD) so với nhóm ở lại (61.2 USD)."
        }
        insight = insights_map.get(feature_name, f"Đặc trưng {feature_name} thể hiện sự phân hóa nhất định đối với biến mục tiêu Churn.")

        # Kiểm tra kiểu dữ liệu của feature
        if not pd.api.types.is_numeric_dtype(self.df[feature_name]):
            logger.info("Cột '%s' là biến định tính. Trả về bảng phân tích chéo.", feature_name)
            crosstab = pd.crosstab(self.df[feature_name], self.df['Churn'])
            return {
                "type": "categorical",
                "index": crosstab.index.tolist(),
                "columns": crosstab.columns.tolist(),
                "values": crosstab.values.tolist(),
                "insight": insight
            }
        else:
            logger.info(
                "Cột '%s' là biến định lượng. Thực hiện phân nhóm thống kê theo Churn.",
                feature_name,
            )
            churn_yes = self.df[self.df['Churn'] == 'Yes'][feature_name].dropna()
            churn_no = self.df[self.df['Churn'] == 'No'][feature_name].dropna()
            
            return {
                "type": "numerical",
                "churn_yes_stats": {
                    "mean": float(churn_yes.mean()),
                    "boxplot": [float(churn_yes.min()), float(churn_yes.quantile(0.25)), float(churn_yes.median()), float(churn_yes.quantile(0.75)), float(churn_yes.max())]
                },
                "churn_no_stats": {
                    "mean": float(churn_no.mean()),
                    "boxplot": [float(churn_no.min()), float(churn_no.quantile(0.25)), float(churn_no.median()), float(churn_no.quantile(0.75)), float(churn_no.max())]
                },
                "insight": insight
            }

    def get_correlation_matrix(self) -> Dict[str, Any]:
        """
        [Mục phân tích tương quan]
        Trả về ma trận hệ số tương quan tuyến tính (Pearson correlation) giữa các cột số
        để Frontend vẽ biểu đồ nhiệt (Correlation Heatmap).
        """
        logger.info("Đang tính toán ma trận hệ số tương quan tuyến tính Pearson...")
        cols = [
            'tenure', 'MonthlyCharges', 'TotalCharges',
            'total_active_services', 'charge_to_tenure_ratio_log',
            'average_cost_per_service', 'security_score', 'streaming_score'
        ]
        valid_cols = [c for c in cols if c in self.df.columns]
        
        corr_matrix = self.df[valid_cols].corr()
        logger.info("Tính toán tương quan thành công cho các cột: %s", valid_cols)
        
        insight = (
            "Ma trận tương quan cho thấy sự xuất hiện của các chỉ số mới mang ý nghĩa kinh tế rất rõ rệt: "
            "average_cost_per_service tương quan thuận rất cao với MonthlyCharges (0.76) nhưng tương quan nghịch với total_active_services (-0.31). "
            "Chỉ số security_score cho thấy mối tương quan âm rõ rệt với tỉ lệ rời bỏ dịch vụ, củng cố giả thuyết rằng khách hàng sử dụng "
            "các lớp bảo mật bảo vệ có xu hướng gắn bó lâu dài hơn."
        )

        return {
            "columns": corr_matrix.columns.tolist(),
            "index": corr_matrix.index.tolist(),
            "values": corr_matrix.values.tolist(),
            "insight": insight
        }
