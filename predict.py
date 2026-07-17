import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("temperature_model.pkl")

st.title("🌡️ Temperature Prediction App")

st.write("Enter weather data to predict temperature.")

# User inputs
city_id = st.number_input("City ID", value=0)
latitude = st.number_input("Latitude", value=52)
longitude = st.number_input("Longitude", value=13)
humidity = st.slider("Humidity (%)", 0, 100, 83)
pressure = st.number_input("Pressure (hPa)", value=1022)
wind_speed = st.number_input("Wind Speed (m/s)", value=1.54)
clouds = st.slider("Cloud Cover (%)", 0, 100, 100)

# Predict button
if st.button("Predict Temperature"):

    data = pd.DataFrame({
        "city_id": [city_id],
        "Latitude": [latitude],
        "Longitude": [longitude],
        "humidity": [humidity],
        "pressure": [pressure],
        "wind_speed": [wind_speed],
        "clouds": [clouds]
    })

    prediction = model.predict(data)

    st.success(f"Predicted Temperature: {prediction[0]:.2f} °C")