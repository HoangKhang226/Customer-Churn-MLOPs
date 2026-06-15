import streamlit as st
import pandas as pd
import requests

def render_predict():
    st.markdown("<h2 style='margin-bottom:0px; margin-top:0px;'>Dự đoán Khách hàng Rời bỏ (Prediction)</h2>", unsafe_allow_html=True)
    st.markdown("<hr style='margin-top:10px; margin-bottom:20px;'/>", unsafe_allow_html=True)
    

    st.markdown("#### Nhập thông tin khách hàng")
    st.markdown("<div style='font-size:14px; color:#555; margin-bottom:15px;'>Điền các thông tin của khách hàng để mô hình LightGBM dự đoán rủi ro rời bỏ.</div>", unsafe_allow_html=True)
    
    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**1. Thông tin Cá nhân**")
            gender = st.selectbox("Giới tính", ["Male", "Female"])
            senior_citizen = st.selectbox("Người cao tuổi", ["No", "Yes"])
            partner = st.selectbox("Có đối tác (Partner)", ["No", "Yes"])
            dependents = st.selectbox("Người phụ thuộc", ["No", "Yes"])
            
        with col2:
            st.markdown("**2. Dịch vụ Đăng ký**")
            internet_service = st.selectbox("Dịch vụ Internet", ["DSL", "Fiber optic", "No"])
            phone_service = st.selectbox("Dịch vụ Điện thoại", ["Yes", "No"])
            online_security = st.selectbox("Bảo mật Trực tuyến", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Hỗ trợ Kỹ thuật", ["No", "Yes", "No internet service"])
            
        with col3:
            st.markdown("**3. Tài khoản & Thanh toán**")
            contract = st.selectbox("Loại hợp đồng", ["Month-to-month", "One year", "Two year"])
            tenure = st.slider("Thời gian sử dụng (Tháng)", 0, 72, 1)
            monthly_charges = st.number_input("Cước hàng tháng ($)", min_value=0.0, max_value=200.0, value=70.0)
            total_charges = st.number_input("Tổng cước ($)", min_value=0.0, max_value=10000.0, value=70.0)
            
        submit_button = st.form_submit_button("Thực hiện Dự đoán", type="primary")
        

    
    if submit_button:
        # Prepare input dictionary
        input_dict = {
            "gender": gender,
            "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": "No", # default
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": "No", # default
            "DeviceProtection": "No", # default
            "TechSupport": tech_support,
            "StreamingTV": "No", # default
            "StreamingMovies": "No", # default
            "Contract": contract,
            "PaperlessBilling": "Yes", # default
            "PaymentMethod": "Electronic check", # default
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "id": 999999, # dummy
        }
        
        try:
            # Call FastAPI Backend
            with st.spinner("Đang xử lý dự đoán từ Backend..."):
                response = requests.post("http://localhost:8000/predict", json=input_dict)
                response.raise_for_status()
                result = response.json()
                prob = result['churn_probability']
            
            st.markdown("#### Kết quả Dự đoán")
            
            res_col1, res_col2 = st.columns([1, 2])
            with res_col1:
                st.markdown("<div style='text-align: center; padding: 20px;'>", unsafe_allow_html=True)
                if prob > 70:
                    color = "#DC3545"
                    status = "Rủi ro Cao"
                    recommend = "Cần liên hệ chăm sóc ngay, đề xuất giảm giá cước hoặc nâng cấp gói hỗ trợ kỹ thuật."
                elif prob > 40:
                    color = "#FFC107"
                    status = "Nguy cơ Trung bình"
                    recommend = "Nên gửi email hoặc tin nhắn giới thiệu lợi ích của hợp đồng dài hạn."
                else:
                    color = "#28A745"
                    status = "An toàn"
                    recommend = "Khách hàng ổn định. Có thể giới thiệu thêm các dịch vụ giải trí (Streaming)."
                    
                st.markdown(f"<h1 style='color:{color}; font-size:48px; margin:0;'>{prob:.1f}%</h1>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:18px; font-weight:bold; color:{color};'>{status}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
            with res_col2:
                st.markdown(f"""
                <div style='background-color: #f8f9fa; border-left: 4px solid {color}; padding: 15px; border-radius: 5px; height: 100%;'>
                    <h5 style='margin-top: 0;'>AI Insights:</h5>
                    <p>Khách hàng có {tenure} tháng sử dụng với hợp đồng <b>{contract}</b> và cước phí <b>${monthly_charges:.2f}</b>/tháng.</p>
                    <p>Mô hình dựa trên các đặc trưng đã biến đổi (feature engineering) như tỷ lệ cước phí/tháng, điểm bảo mật, và phân khúc nhân khẩu học để đưa ra quyết định.</p>
                    <b>Hành động Đề xuất:</b> {recommend}
                </div>
                """, unsafe_allow_html=True)
                

            
        except Exception as e:
            st.error(f"Prediction Pipeline Error: {e}")
