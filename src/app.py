import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import pickle

st.set_page_config(page_title="Glucose Predictor", layout="centered")

st.title("🧠 Glucose Prediction System")
st.write("Predict future glucose level using past readings")

# -----------------------------
# LOAD MODEL + SCALER
# -----------------------------
model = load_model("src/model.h5", compile=False)

with open("src/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


# -----------------------------
# USER INPUT
# -----------------------------
st.subheader("Enter last 10 glucose readings")

user_input = st.text_input(
    "Comma separated values",
    "120,118,115,113,110,108,107,105,104,102"
)

if st.button("Predict"):

    try:
        values = list(map(float, user_input.split(",")))

        if len(values) != 10:
            st.error("Please enter exactly 10 values")
        else:
            arr = np.array(values).reshape(-1,1)

            arr = scaler.transform(arr)

            arr = arr.reshape(1, 10, 1)

            prediction = model.predict(arr)

            prediction = scaler.inverse_transform(prediction)

            st.success(f"Predicted Glucose: {prediction[0][0]:.2f}")

    except:
        st.error("Invalid input format")

#🔥 IMPORTANT (SAVE MODEL + SCALER)

import pickle

model.save("model.h5")

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

