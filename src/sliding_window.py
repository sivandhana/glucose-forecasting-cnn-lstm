import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("cgm_dataset.txt", sep="|")

data["DataDtTm"] = pd.to_datetime(
    data["DataDtTm"],
    format="%d%b%y:%H:%M:%S"
)

data = data.sort_values("DataDtTm")

#extracting glucose values
glucose_values = data["CGM"].values

#sliding window function
def create_sliding_windows(series, window_size):

    X = []
    y = []

    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])

    return np.array(X), np.array(y)

window_size = 5

X, y = create_sliding_windows(glucose_values, window_size)

print("\nExample Sliding Windows:\n")

for i in range(5):
    print("Input sequence:", X[i])
    print("Target value:", y[i])
    print()

print("Input shape:", X.shape)
print("Target shape:", y.shape)

#shaping for cnn-lstm model (laterrr lol)

#cnn/lstm models require 3d input

#shape format(samples, timesteps, features)

X = X.reshape((X.shape[0], X.shape[1], 1))

print("CNN/LSTM input shape:", X.shape)

#GRAPHSSSS MWAHAHAHA

#1. visualising a sliding windoww

import matplotlib.pyplot as plt

plt.plot(X[0])
plt.title("Example Sliding Window")
plt.xlabel("Time Step")
plt.ylabel("Glucose")

#plt.show()