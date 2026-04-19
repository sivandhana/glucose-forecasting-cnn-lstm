import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

data = pd.read_csv("cgm_dataset.txt", sep="|")

data["DataDtTm"] = pd.to_datetime(
    data["DataDtTm"],
    format="%d%b%y:%H:%M:%S"
)

data = data.sort_values("DataDtTm")

glucose = data["CGM"].values

# reduce dataset size to avoid memory issues
glucose = glucose[:5000]
glucose = glucose.astype(np.float32)

#sliding window
def create_windows(series, window):

    X = []
    y = []

    for i in range(len(series) - window):
        X.append(series[i:i+window])
        y.append(series[i+window])

    return np.array(X), np.array(y)

#comparison between 5,10,15

window_sizes = [5, 10, 15]

results = {}

for w in window_sizes:

    X, y = create_windows(glucose, w)

    split = int(len(X) * 0.8)

    X_train = X[:split]
    X_test = X[split:]

    y_train = y[:split]
    y_test = y[split:]

    model = RandomForestRegressor(
        n_estimators=20,
        max_depth=5,
        n_jobs=1,
        random_state=42
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, pred))

    results[w] = rmse

    print(f"Window {w} → RMSE: {rmse}")

#plotting

plt.figure(figsize=(8,5))

windows = list(results.keys())
errors = list(results.values())

bars = plt.bar(windows, errors)

# highlight best window
best_window = min(results, key=results.get)

for bar, w in zip(bars, windows):
    if w == best_window:
        bar.set_edgecolor('black')
        bar.set_linewidth(2)

# add value labels on top
for i, v in enumerate(errors):
    plt.text(windows[i], v + 0.2, f"{v:.2f}", ha='center')

plt.xlabel("Sliding Window Size (time steps)")
plt.ylabel("RMSE Error")
plt.title("Effect of Sliding Window Size on Glucose Prediction")

plt.xticks(windows)

plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.show()

print("\nFinal Comparison:")

for w, r in results.items():
    print(f"Window size {w}: RMSE = {r}")
best_window = min(results, key=results.get)

print(f"\nBest window size: {best_window} (lowest RMSE)")