import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Glucose Forecasting System", layout="centered")

st.title("Glucose Forecasting System")
st.markdown("Forecast future glucose levels from the most recent 10 readings.")

# -----------------------------
# LOAD MODEL AND SCALER
# -----------------------------
model = load_model("model.h5", compile=False)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# -----------------------------
# USER INPUT
# -----------------------------
st.subheader("Enter Last 10 Glucose Readings")
user_input = st.text_input(
    "Comma-separated glucose values (mg/dL)",
    "120,118,115,113,110,108,107,105,104,102"
)

forecast_steps = st.slider("Select forecast horizon (future steps)", 1, 10, 5)

# -----------------------------
# FORECAST FUNCTION
# -----------------------------
def multi_step_forecast(model, input_values, scaler, steps):
    """
    input_values: list of 10 glucose values in original scale
    steps: number of future values to predict
    """
    history = np.array(input_values, dtype=np.float32).reshape(-1, 1)

    # scale the initial history
    history_scaled = scaler.transform(history).flatten().tolist()

    predictions = []

    for _ in range(steps):
        current_window = np.array(history_scaled[-10:]).reshape(1, 10, 1)

        pred_scaled = model.predict(current_window, verbose=0)
        pred_value = scaler.inverse_transform(pred_scaled)[0][0]

        predictions.append(pred_value)

        # append predicted scaled value back into history
        history_scaled.append(pred_scaled[0][0])

    return predictions

# -----------------------------
# BUTTON ACTION
# -----------------------------
if st.button(" Generate Forecast"):
    try:
        values = [float(x.strip()) for x in user_input.split(",")]

        if len(values) != 10:
            st.error("Please enter exactly 10 glucose readings.")
        else:
            predictions = multi_step_forecast(
                model=model,
                input_values=values,
                scaler=scaler,
                steps=forecast_steps
            )

            st.subheader("Forecast Results")
            st.write(f"**Next predicted glucose value:** {predictions[0]:.2f} mg/dL")

            # risk interpretation for first prediction
            first_pred = predictions[0]
            if first_pred < 70:
                st.warning("⚠️ Risk Alert: Predicted glucose is in hypoglycemic range.")
            elif first_pred > 180:
                st.warning("⚠️ Risk Alert: Predicted glucose is in hyperglycemic range.")
            else:
                st.success("✅ Predicted glucose is within a normal/acceptable range.")

            # Show forecast table
            st.subheader("Predicted Future Values")
            for i, pred in enumerate(predictions, start=1):
                st.write(f"Step {i}: {pred:.2f} mg/dL")

            # -----------------------------
            # PLOT GRAPH
            # -----------------------------
            history_x = list(range(1, 11))
            forecast_x = list(range(10, 10 + forecast_steps + 1))

            plot_values = [values[-1]] + predictions

            fig, ax = plt.subplots(figsize=(10, 5))

            # history
            ax.plot(history_x, values, marker='o', linewidth=2, label="Past Glucose Readings")

            # forecast
            ax.plot(forecast_x, plot_values, marker='o', linestyle='--', linewidth=2, label="Forecasted Glucose")

            # separator
            ax.axvline(x=10, color='red', linestyle=':', linewidth=2, label="Forecast Start")

            ax.set_title("Glucose Forecast Graph")
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Glucose (mg/dL)")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.5)

            st.pyplot(fig)

    except ValueError:
        st.error("Invalid input. Please enter only numbers separated by commas.")