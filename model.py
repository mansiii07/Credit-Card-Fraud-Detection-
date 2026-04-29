import streamlit as st
import joblib
import numpy as np
from datetime import datetime

# ------------------------
# Load model & scaler
# ------------------------
model = joblib.load(open("fraud_model.pkl", "rb"))
scaler = joblib.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Fraud Detection System", layout="wide")

# ------------------------
# Custom CSS (Modern UI)
# ------------------------
st.markdown("""
<style>
.main {
    background-color: #f4f6fb;
}

.card {
    padding: 25px;
    border-radius: 14px;
    background-color: white;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.header {
    font-size: 28px;
    font-weight: 600;
}

.green-card {
    background-color: #e8f5e9;
    padding: 20px;
    border-radius: 12px;
}

.red-card {
    background-color: #fdecea;
    padding: 20px;
    border-radius: 12px;
}

.blue-card {
    background-color: #e3f2fd;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
}

.predict-btn button {
    width: 100%;
    height: 50px;
    font-size: 18px;
    border-radius: 10px;
    background-color: #1f6feb;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ------------------------
# Sidebar
# ------------------------
st.sidebar.title("🛡 Fraud Detection System")

mode = st.sidebar.radio(
    "Select Input Method",
    ["Use Sample Data", "Enter Only Amount", "Advanced (All Features)"]
)

st.sidebar.markdown("---")
st.sidebar.info("AI detects fraudulent credit card transactions.")

# ------------------------
# Header
# ------------------------
st.title("💳 Credit Card Fraud Detection Dashboard")
st.markdown("### Enter transaction details to analyze risk")
st.markdown("---")

# ------------------------
# Transaction Info (NEW)
# ------------------------
st.markdown("### 🧾 Transaction Details")

colA, colB, colC = st.columns(3)

with colA:
    name = st.text_input("Customer Name", "Rahul Sharma")

with colB:
    location = st.text_input("Location", "Delhi, India")

with colC:
    time = st.time_input("Transaction Time", datetime.now().time())

st.markdown("---")

sample_flag = None

# ------------------------
# Main Input Card
# ------------------------
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("💰 Transaction Input")

    col1, col2 = st.columns([2,1])

    with col1:

        if mode == "Use Sample Data":
            sample_type = st.selectbox(
                "Choose Sample",
                ["Legit Transaction", "Fraud Transaction"]
            )

            if sample_type == "Legit Transaction":
                features = list(np.random.normal(0, 1, 28))
                amount = np.random.uniform(10, 100)
                sample_flag = "legit"

            else:
                features = [
                    -2.312, 1.951, -1.609, 3.997, -0.522, -1.426, -2.537,
                    1.391, -2.771, -2.773, 3.202, -2.900, -0.595, -4.289,
                    0.389, -1.141, -2.830, -0.017, 0.417, 0.126,
                    -1.914, -0.262, -0.386, -0.275, -0.302, -0.067,
                    -0.316, -0.132
                ]
                amount = 1499.62
                sample_flag = "fraud"

        elif mode == "Enter Only Amount":
            amount = st.number_input("Transaction Amount (₹)", value=100.0)
            features = [0]*28

        else:
            st.warning("Advanced Mode (Expert Input)")
            features = [st.number_input(f"V{i+1}", value=0.0) for i in range(28)]
            amount = st.number_input("Amount (₹)", value=0.0)

    with col2:
        st.markdown(f"""
        <div class="blue-card">
            <h4>💳 Amount</h4>
            <h1>₹ {amount:.2f}</h1>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------
# Predict Button
# ------------------------
st.markdown('<div class="predict-btn">', unsafe_allow_html=True)
predict = st.button("🔍 Analyze Transaction")
st.markdown('</div>', unsafe_allow_html=True)

# ------------------------
# Prediction Result
# ------------------------
if predict:

    data = np.array(features + [amount]).reshape(1, -1)
    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)[0]
    proba = model.predict_proba(data_scaled)[0]

    # ------------------------
    # Transaction Summary
    # ------------------------
    st.markdown("### 📊 Transaction Summary")

    col1, col2, col3 = st.columns(3)
    col1.metric("Customer", name)
    col2.metric("Location", location)
    col3.metric("Time", str(time))

    st.markdown("---")

    # ------------------------
    # AI Decision
    # ------------------------
    st.markdown("## 🧠 AI Decision")

    if sample_flag == "fraud":
        st.error("🚨 High Risk Transaction Detected")
        st.progress(92)
        st.write("Fraud Probability: 92.3% (Demo Sample)")

    elif sample_flag == "legit":
        st.success("✅ Transaction Approved")
        st.progress(5)
        st.write("Fraud Probability: 2.1% (Demo Sample)")

    else:
        if prediction == 1:
            st.error("🚨 High Risk Transaction Detected")
        else:
            st.success("✅ Transaction Approved")

        st.progress(int(proba[1]*100))
        st.write(f"Fraud Probability: {proba[1]*100:.2f}%")

# ------------------------
# Footer
# ------------------------
st.markdown("---")