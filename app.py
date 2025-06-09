import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load mô hình
with open("BIDV_model.pkl", "rb") as f:
    model = pickle.load(f)

# Giao diện ứng dụng
st.title("Dự đoán khả năng cho vay của khách hàng - BIDV")

# Nhập liệu
gender = st.selectbox("Giới tính", ["Nam", "Nữ"])
married = st.selectbox("Tình trạng hôn nhân", ["Độc thân", "Đã kết hôn"])
dependents = st.selectbox("Số người phụ thuộc", ["0", "1", "2", "3+"])
education = st.selectbox("Trình độ học vấn", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Tự kinh doanh?", ["Có", "Không"])
applicant_income = st.number_input("Thu nhập người vay", min_value=0)
coapplicant_income = st.number_input("Thu nhập người cùng vay", min_value=0)
loan_amount = st.number_input("Số tiền vay (nghìn)", min_value=0)
loan_term = st.number_input("Thời hạn vay (tháng)", min_value=1)
credit_history = st.selectbox("Lịch sử tín dụng", ["Tốt", "Xấu"])
property_area = st.selectbox("Khu vực bất động sản", ["Urban", "Semiurban", "Rural"])

# Mã hóa dữ liệu đầu vào
gender = 1 if gender == "Nam" else 0
married = 1 if married == "Đã kết hôn" else 0
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Có" else 0
credit_history = 1 if credit_history == "Tốt" else 0
dependents = 3 if dependents == "3+" else int(dependents)

# Gộp thành mảng đầu vào
input_data = np.array([[gender, married, dependents, education, self_employed,
                        applicant_income, coapplicant_income, loan_amount, loan_term,
                        credit_history, 0, 0, 0]])

# Mã hóa One-hot cho Property_Area
if property_area == "Urban":
    input_data[0, 10] = 1
elif property_area == "Semiurban":
    input_data[0, 11] = 1
else:
    input_data[0, 12] = 1

# Dự đoán
if st.button("Dự đoán"):
    result = model.predict(input_data)
    if result[0] == 1:
        st.success("✅ Khách hàng CÓ khả năng được vay!")
    else:
        st.error("❌ Khách hàng KHÔNG được vay.")
