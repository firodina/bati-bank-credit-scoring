import requests
import streamlit as st

# ============================================================
# CONFIG
# ============================================================
API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="💳",
    layout="centered"
)

st.title("💳 Credit Risk Predictor")
st.markdown("Enter transaction details to predict **credit risk**")
st.markdown("---")

# ============================================================
# INPUT FORM (MATCHES MODEL)
# ============================================================
col1, col2 = st.columns(2)

with col1:
    country_code = st.number_input(
        "Country Code",
        min_value=1,
        step=1,
        value=256,
        help="Numeric country identifier"
    )

    pricing_strategy = st.selectbox(
        "Pricing Strategy",
        options=[0, 1, 2, 3],
        help="Encoded pricing strategy"
    )

    currency_code_ugx = st.selectbox(
        "Currency Code UGX",
        options=[0, 1],
        help="1 = UGX, 0 = Other"
    )

with col2:
    amount = st.number_input(
        "Amount",
        min_value=0.0,
        step=100.0,
        value=1000.0
    )

    value = st.number_input(
        "Value",
        min_value=0.0,
        step=100.0,
        value=1000.0
    )

# ============================================================
# PREDICT BUTTON
# ============================================================
st.markdown("---")

if st.button("🔮 Predict Risk", use_container_width=True):

    payload = {
        "CountryCode": int(country_code),
        "Amount": float(amount),
        "Value": float(value),
        "PricingStrategy": int(pricing_strategy),
        "CurrencyCode_UGX": int(currency_code_ugx)
    }

    with st.spinner("Calling FastAPI model..."):
        try:
            response = requests.post(API_URL, json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()

            st.markdown("### 📊 Prediction Result")

            st.metric(
                label="Default Probability",
                value=f"{result['default_probability']:.2%}"
            )

            if result["risk_label"] == "high":
                st.error("⚠️ HIGH RISK")
            else:
                st.success("✅ LOW RISK")

            with st.expander("Request Payload"):
                st.json(payload)

            with st.expander("API Response"):
                st.json(result)

        except requests.exceptions.ConnectionError:
            st.error("❌ FastAPI is not running on http://localhost:8000")
        except Exception as e:
            st.error(f"❌ Error: {e}")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption("Streamlit UI → FastAPI → MLflow Model")
