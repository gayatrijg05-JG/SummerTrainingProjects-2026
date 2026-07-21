import pickle
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Used Car Price Prediction", page_icon="🚗", layout="centered")

model_path = Path(__file__).with_name("UsedCarMODEL.pkl")
model = None

try:
    with model_path.open("rb") as f:
        model = pickle.load(f)
except Exception:
    st.error("Failed to load model. Place UsedCarMODEL.pkl next to app.py.")


def predict_price(model, yr_mfr, kms_run, fuel, transmission, body, make, owners, broker_quote):
    data = pd.DataFrame([
        {
            "yr_mfr": yr_mfr,
            "kms_run": kms_run,
            "total_owners": owners,
            "broker_quote": broker_quote,
            "fuel_type": fuel.lower(),
            "body_type": body.lower(),
            "transmission": transmission.lower(),
            "make": make.lower(),
        }
    ])
    data = pd.get_dummies(data, columns=["fuel_type", "body_type", "transmission", "make"])
    data = data.reindex(columns=model.feature_names_in_, fill_value=0)
    return float(model.predict(data)[0])


st.title("🚗 Used Car Price Prediction")
st.write("Enter the car details below and click Predict.")

yr_mfr = st.number_input("Manufacturing Year", min_value=2000, max_value=2026, value=2020)
kms_run = st.number_input("Kilometers Driven", min_value=0, value=50000)

fuel = st.selectbox("Fuel Type", ["Diesel", "Petrol", "Electric", "Petrol & CNG", "Petrol & LPG"])
transmission = st.selectbox("Transmission", ["Automatic", "Manual"])
body = st.selectbox("Body Type", ["SUV", "Sedan", "Hatchback", "Luxury SUV", "Luxury Sedan"])
make = st.selectbox("Car Brand", ["Maruti", "Hyundai", "Honda", "Toyota", "Mahindra", "Tata", "Ford", "BMW", "Audi", "Chevrolet", "Jeep", "Kia", "Mercedes Benz", "MG", "Nissan", "Renault", "Skoda", "Volkswagen", "Volvo"])
owners = st.selectbox("Number of Owners", [1, 2, 3, 4, 5])
broker_quote = st.number_input("Broker Quote (₹)", min_value=10000, value=500000)

if st.button("Predict Price"):
    if model is None:
        st.error("Model not available.")
    else:
        price = predict_price(
            model,
            yr_mfr=yr_mfr,
            kms_run=kms_run,
            fuel=fuel,
            transmission=transmission,
            body=body,
            make=make,
            owners=owners,
            broker_quote=broker_quote,
        )
        st.subheader("Predicted Selling Price")
        st.success(f"₹{price:,.0f}")

st.caption("Developed by Gayatri")
st.caption("Summer Training Project | Streamlit")
