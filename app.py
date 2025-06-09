# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Đọc dữ liệu
df = pd.read_csv('loan.csv')

# Xử lý dữ liệu thiếu
df.fillna({
    'Gender': df['Gender'].mode()[0],
    'Married': df['Married'].mode()[0],
    'Dependents': df['Dependents'].mode()[0],
    'Self_Employed': df['Self_Employed'].mode()[0],
    'LoanAmount': df['LoanAmount'].mean(),
    'Loan_Amount_Term': df['Loan_Amount_Term'].mean(),
    'Credit_History': df['Credit_History'].mode()[0]
}, inplace=True)

# Mã hóa các biến phân loại
le = LabelEncoder()
for col in ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']:
    df[col] = le.fit_transform(df[col])

# Xử lý biến phụ thuộc
df['Dependents'].replace('3+', 3, inplace=True)
df['Dependents'] = df['Dependents'].astype(int)

# Chọn đặc trưng đầu vào và biến mục tiêu
X = df.drop(['Loan_ID', 'Loan_Status'], axis=1)
y = df['Loan_Status']

# Tách tập train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Huấn luyện mô hình
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Lưu mô hình
joblib.dump(model, 'loan_model.pkl')
import streamlit as st
import joblib
import numpy as np

# Load mô hình
model = joblib.load("loan_model.pkl")

st.title("Ứng dụng dự đoán khả năng cho vay của ngân hàng BIDV")

# Nhập thông tin
gender = st.selectbox("Giới tính", ["Nam", "Nữ"])
married = st.selectbox("Tình trạng hôn nhân", ["Độc thân", "Đã kết hôn"])
dependents = st.selectbox("Số người phụ thuộc", [0, 1, 2, 3])
education = st.selectbox("Trình độ học vấn", ["Tốt nghiệp", "Chưa tốt nghiệp"])
self_employed = st.selectbox("Tự kinh doanh", ["Không", "Có"])
applicant_income = st.number_input("Thu nhập người vay (VND)", min_value=1)
coapplicant_income = st.number_input("Thu nhập người đồng vay (VND)", min_value=1)
loan_amount = st.number_input("Số tiền vay (VND)", min_value=1)
loan_term = st.number_input("Thời hạn vay (tháng)", min_value=1)
credit_history = st.selectbox("Lịch sử tín dụng tốt?", ["Có", "Không"])
property_area = st.selectbox("Khu vực tài sản", ["Đô thị", "Nông thôn", "Ngoại thành"])

# Map các giá trị nhập thành số
gender = 1 if gender == "Nam" else 0
married = 1 if married == "Đã kết hôn" else 0
education = 0 if education == "Tốt nghiệp" else 1
self_employed = 1 if self_employed == "Có" else 0
credit_history = 1.0 if credit_history == "Có" else 0.0
property_map = {"Đô thị": 2, "Ngoại thành": 1, "Nông thôn": 0}
property_area = property_map[property_area]

# Khi bấm nút "Kiểm tra"
if st.button("Kiểm tra"):
    input_data = np.array([[gender, married, dependents, education, self_employed,
                            applicant_income, coapplicant_income, loan_amount,
                            loan_term, credit_history, property_area]])
    prediction = model.predict(input_data)[0]
    result = "✅ Hồ sơ của khách hàng đã đáp ứng đủ yêu cầu nên được duyệt vay" if prediction == 1 else "❌ Xin lỗi! Hồ sơ của khách hàng chưa đáp ứng đủ yêu cầu nên không được duyệt khoản vay"
    st.success(result)
    st.write(result)
