import pathlib
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# ------------------------------------------------
# Load trained Random Forest pipeline
# ------------------------------------------------

MODEL_PATH = pathlib.Path(__file__).parent / "model.pkl"

try:
    model = joblib.load(MODEL_PATH)
    model_loaded = True
    load_error = ""
except Exception as e:
    # Fallback dummy model so the UI still runs if loading fails
    class DummyModel:
        def predict(self, X):
            return np.array([0])
        def predict_proba(self, X):
            return np.array([[0.8, 0.2]])  # 20 percent cancel

    model = DummyModel()
    model_loaded = False
    load_error = str(e)

# ------------------------------------------------
# Categorical options
# ------------------------------------------------

DEPOSIT_TYPES = [
    "No Deposit",
    "Non Refund",
    "Refundable",
]

MARKET_SEGMENTS = [
    "Online TA",
    "Offline TA/TO",
    "Direct",
    "Corporate",
    "Complementary",
    "Groups",
    "Aviation",
]

CUSTOMER_TYPES = [
    "Transient",
    "Contract",
    "Transient-Party",
    "Group",
]

# ------------------------------------------------
# Streamlit UI
# ------------------------------------------------

st.set_page_config(
    page_title="Hotel Cancellation Risk",
    page_icon="🏨",
    layout="centered",
)

st.title("🏨 Hotel Booking Cancellation Risk")

st.write(
    "This app uses a Random Forest model trained on historical bookings "
    "to estimate whether a new reservation is likely to be cancelled."
)

with st.form("booking_form"):
    st.subheader("Booking details")

    col1, col2 = st.columns(2)

    with col1:
        lead_time = st.number_input(
            "Lead time (days between booking and arrival)",
            min_value=0,
            max_value=365 * 2,
            value=60,
        )
        stays_weekend = st.number_input(
            "Weekend nights",
            min_value=0,
            max_value=14,
            value=2,
        )
        stays_week = st.number_input(
            "Week nights",
            min_value=0,
            max_value=21,
            value=3,
        )
        adr = st.number_input(
            "Average daily rate (ADR)",
            min_value=0.0,
            max_value=1000.0,
            value=120.0,
            step=1.0,
        )
        prev_cancels = st.number_input(
            "Previous cancellations",
            min_value=0,
            max_value=20,
            value=0,
        )

    with col2:
        special_requests = st.number_input(
            "Total special requests",
            min_value=0,
            max_value=10,
            value=1,
        )
        car_spaces = st.number_input(
            "Required car parking spaces",
            min_value=0,
            max_value=5,
            value=0,
        )
        adults = st.number_input(
            "Adults",
            min_value=1,
            max_value=10,
            value=2,
        )
        children = st.number_input(
            "Children",
            min_value=0,
            max_value=10,
            value=0,
        )
        babies = st.number_input(
            "Babies",
            min_value=0,
            max_value=5,
            value=0,
        )

    st.subheader("Customer and channel")

    col3, col4, col5 = st.columns(3)

    with col3:
        deposit_type = st.selectbox("Deposit type", DEPOSIT_TYPES)
    with col4:
        market_segment = st.selectbox("Market segment", MARKET_SEGMENTS)
    with col5:
        customer_type = st.selectbox("Customer type", CUSTOMER_TYPES)

    submitted = st.form_submit_button("Predict cancellation risk")

if submitted:
    # Build a one row DataFrame with the exact feature names
    input_dict = {
        "lead_time": lead_time,
        "adr": adr,
        "stays_in_weekend_nights": stays_weekend,
        "stays_in_week_nights": stays_week,
        "previous_cancellations": prev_cancels,
        "total_of_special_requests": special_requests,
        "required_car_parking_spaces": car_spaces,
        "adults": adults,
        "children": children,
        "babies": babies,
        "deposit_type": deposit_type,
        "market_segment": market_segment,
        "customer_type": customer_type,
    }

    input_df = pd.DataFrame([input_dict])

    pred_class = int(model.predict(input_df)[0])

    # Assumes column 1 is the probability of cancellation
    try:
        proba = model.predict_proba(input_df)[0]
        cancel_prob = float(proba[1])
    except Exception:
        cancel_prob = 0.2  # fallback if dummy model

    cancel_prob_pct = cancel_prob * 100

    st.subheader("Prediction")

    if pred_class == 1:
        st.write("🔴 This booking is **likely to be cancelled**.")
    else:
        st.write("🟢 This booking is **likely to be honored**.")

    st.metric(
        "Estimated cancellation probability",
        f"{cancel_prob_pct:0.1f} %",
    )

    if not model_loaded:
        st.caption(
            "Demo note, the real model file could not be loaded in this environment. "
            "This run is using a placeholder model so I can still demonstrate the app flow."
        )
        st.caption(f"Load error: {load_error}")
    else:
        st.caption(
            "Model, Random Forest trained on the hotel_bookings dataset using 13 business friendly features."
        )
