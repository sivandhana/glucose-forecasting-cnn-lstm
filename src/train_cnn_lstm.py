import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout


# -----------------------------
# 1. LOAD DATA
# -----------------------------
data = pd.read_csv("cgm_dataset.txt", sep="|")

data["DataDtTm"] = pd.to_datetime(
    data["DataDtTm"],
    format="%d%b%y:%H:%M:%S"
)

data = data.sort_values("DataDtTm")

# 🔻 reduce dataset (important for your laptop)
data = data[:3000]


# -----------------------------
# 2. NORMALIZE
# -----------------------------
scaler = MinMaxScaler()

data["CGM"] = scaler.fit_transform(data[["CGM"]])


# -----------------------------
# 3. CREATE SLIDING WINDOWS
# -----------------------------
def create_windows(series, window):

    X = []
    y = []

    for i in range(len(series) - window):
        X.append(series[i:i+window])
        y.append(series[i+window])

    return np.array(X), np.array(y)


window_size = 10  # best from your experiment

glucose = data["CGM"].values

X, y = create_windows(glucose, window_size)


# -----------------------------
# 4. RESHAPE FOR CNN-LSTM
# -----------------------------
X = X.reshape((X.shape[0], X.shape[1], 1))


# -----------------------------
# 5. TRAIN TEST SPLIT
# -----------------------------
split = int(len(X) * 0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


# -----------------------------
# 6. BUILD MODEL (LIGHTWEIGHT)
# -----------------------------
model = Sequential()

model.add(Conv1D(
    filters=16,
    kernel_size=2,
    activation='relu',
    input_shape=(window_size, 1)
))

model.add(LSTM(20))

model.add(Dropout(0.2))

model.add(Dense(1))

model.compile(
    optimizer='adam',
    loss='mse'
)

model.summary()


# -----------------------------
# 7. TRAIN MODEL
# -----------------------------
history = model.fit(
    X_train,
    y_train,
    epochs=5,          # keep small
    batch_size=16,
    validation_split=0.2
)


# -----------------------------
# 8. PREDICTION
# -----------------------------
pred = model.predict(X_test)

# inverse scaling
pred = scaler.inverse_transform(pred)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1,1))


# -----------------------------
# 9. RMSE
# -----------------------------
rmse = np.sqrt(mean_squared_error(y_test_actual, pred))

print("RMSE:", rmse)


# -----------------------------
# 10. PLOT RESULTS
# -----------------------------
plt.figure(figsize=(10,5))

plt.plot(y_test_actual[:200], label="Actual")
plt.plot(pred[:200], label="Predicted")

plt.title("CNN-LSTM Glucose Prediction")
plt.xlabel("Time")
plt.ylabel("Glucose")

plt.legend()
plt.show()



model.save("model.h5")

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Saved model and scaler")