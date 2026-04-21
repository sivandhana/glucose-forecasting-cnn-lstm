import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Glucose Forecasting System", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background-color: #F8FBFD;
        color: #203040;
    }

    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #203040;
        margin-bottom: 0.2rem;
    }

    .sub-title {
        font-size: 1rem;
        color: #5B7083;
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #2D6CDF;
        margin-top: 1rem;
        margin-bottom: 0.6rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #EAF2FF 0%, #E5F7F2 100%);
        padding: 1rem 1.2rem;
        border-radius: 14px;
        border: 1px solid #D6E2EE;
        box-shadow: 0 4px 14px rgba(32, 48, 64, 0.06);
        margin-bottom: 1rem;
    }

    .metric-label {
        font-size: 0.9rem;
        color: #5B7083;
        margin-bottom: 0.25rem;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #203040;
    }

    .panel {
        background: #FFFFFF;
        padding: 1rem 1.2rem;
        border-radius: 14px;
        border: 1px solid #D6E2EE;
        box-shadow: 0 4px 14px rgba(32, 48, 64, 0.05);
        margin-bottom: 1rem;
    }

    .status-good {
        background-color: #E8F7F1;
        color: #1D6B57;
        padding: 0.9rem 1rem;
        border-radius: 12px;
        border: 1px solid #BFE7DA;
        font-weight: 600;
    }

    .status-warn {
        background-color: #FFF4E5;
        color: #9A6700;
        padding: 0.9rem 1rem;
        border-radius: 12px;
        border: 1px solid #F1D3A8;
        font-weight: 600;
    }

    .status-risk {
        background-color: #FDECEC;
        color: #A33A3A;
        padding: 0.9rem 1rem;
        border-radius: 12px;
        border: 1px solid #F2C4C4;
        font-weight: 600;
    }

    div.stButton > button {
        background: linear-gradient(90deg, #2D6CDF 0%, #2FAE8F 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.65rem 1.2rem;
        font-weight: 600;
        width: 100%;
    }

    div.stButton > button:hover {
        filter: brightness(1.05);
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Glucose Forecasting System", layout="centered")

st.markdown('<div class="main-title">Glucose Forecasting System</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="sub-title">Forecast future glucose levels from recent continuous glucose readings using a CNN-LSTM model.</div>',
    unsafe_allow_html=True
)

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
left_col, right_col = st.columns([1, 1])
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
                st.markdown(
                    '<div class="status-risk">Predicted glucose is in the hypoglycemic range.</div>',
                    unsafe_allow_html=True
                )
            elif first_pred > 180:
                st.markdown(
                    '<div class="status-warn">Predicted glucose is in the hyperglycemic range.</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="status-good">Predicted glucose is within an acceptable range.</div>',
                    unsafe_allow_html=True
                )

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
            fig.patch.set_facecolor("#F8FBFD")
            ax.set_facecolor("#FFFFFF")

            # past readings
            ax.plot(
                history_x,
                values,
                marker='o',
                linewidth=2.5,
                markersize=7,
                color="#2D6CDF",
                label="Past Glucose Readings"
            )

            # forecast
            ax.plot(
                forecast_x,
                plot_values,
                marker='o',
                linewidth=2.5,
                linestyle='--',
                markersize=7,
                color="#2FAE8F",
                label="Forecasted Glucose"
            )

            # separator
            ax.axvline(
                x=10,
                color="#7AA5E8",
                linestyle=':',
                linewidth=2,
                label="Forecast Start"
            )

            ax.set_title("Glucose Forecast", fontsize=16, color="#203040", pad=14)
            ax.set_xlabel("Time Step", fontsize=12, color="#203040")
            ax.set_ylabel("Glucose (mg/dL)", fontsize=12, color="#203040")

            ax.tick_params(colors="#4F6475")
            ax.grid(True, linestyle='--', alpha=0.25, color="#AFC6D9")

            for spine in ax.spines.values():
                spine.set_color("#D6E2EE")

            legend = ax.legend(frameon=True, facecolor="white", edgecolor="#D6E2EE")
            for text in legend.get_texts():
                text.set_color("#203040")

            plt.tight_layout()
            st.pyplot(fig)

    except ValueError:
        st.error("Invalid input. Please enter only numbers separated by commas.")