# la_features

**Phần 6: Contract, Internet & Payment:**

**is_month_to_month = 1**: 29.918 dòng, tỷ lệ churn 42.5%

**is_fiber_optic = 1**: 272.386 dòng, tỷ lệ churn là 41.54%

**is_electronic_check **= 1: 215.372 dòng, tỷ lệ churn là 48.91%

Lí do:

Thực tiễn kinh doanh: 3 điểm về mặt trải nghiệm khách hàng và ràng buộc pháp lý. Hợp đồng tháng k có phí phạt huỷ hợp đồng. Cáp quang tốc độ k cao nhưng có cước đắt và dễ gặp sự cố đường truyền hơn cáp đồng. Thanh toán séc điện tử yêu cầu khách phải thao tác chủ động mỗi tháng, nên họ nhìn được hoá đơn và cân nhắc huỷ dịch vụ

Giữ các biến nhị phân dễ xây dựng mô hình như Decision Tree, Logistic Regression

**Phần 7: Tenure, Charges & Churn:**

**high_risk_profile:** 148.153 khách, tỷ lệ churn 61.23% (cao gấp 3 lần trung bình)

Gom chính xác phân khúc khachs nhạy cảm. Giúp mô hình phân tách nhóm rủi ro cực đoan

**is_low_tenure:** 152.646 dòng, churn 49.39% (gần phân nửa khách hàng mới trong năm đầu)

Năm đầu là giai đoạn khách chưa quen hệ thống hoặc đăng ký để lấy khuyến mãi. Giúp mô hình tập trung phát hiện hành vi nhóm khách chưa ổn định.

**charge_per_tenure:** tương quan dương mạnh với churn (0.2378). Lấy TotalCharges/tenure, biến đo chi phí thực tế trung bình mỗi tháng. Người có tốc độ chi tiêu cao thường nhạy cảm về giá và dễ churn khi hết ưu đãi.

**Phần 8: Service Count & Churn:**

**service_count:**

Khách dùng 1 dịch vụ: 2.67%

Khách dùng 2 dịch vụ: 23.38% > baseline

Khách dùng 3 dịch vụ: 41.32%, đạt đỉnh rủi ro

Khách dùng từ 6 dịch vụ trở lên: 28.27% -> 2.08%, giảm mạnh

=> Quy luật phi tuyến tính: dùng ít dịch vụ, gói cơ bản và rẻ tiền nên k cần huỷ. Dùng đến 3 dịch vụ, bill tăng lên nhưng stickiness với nhà mạng chưa cao nên dễ churn. Dùng trên 6 dịch vụ, khách gắn bó tin tưởng nên k muốn đổi sang nhà mạng khác -> khách hàng trung thành. Giữ feature này sẽ cho thấy độ gắn kết của khách