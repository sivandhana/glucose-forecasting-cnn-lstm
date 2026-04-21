import pandas as pd
import matplotlib.pyplot as plt

# load dataset
data = pd.read_csv("data/cgm_dataset.txt", sep="|")

# display first rows
print("First 10 rows of dataset:")
print(data.head(10))

#Data structure
print("\nDataset Info:")
print(data.info())

print("\nSummary Statistics:")
print(data.describe())

#Missing values
print("\nMissing values per column:")
print(data.isnull().sum())

#number of patients
print("\nUnique Patients:")
print(data["PtID"].unique())

print("\nNumber of patients:")
print(data["PtID"].nunique())

#PREEE PROCESSING LOL

#converting time stamp
data["DataDtTm"] = pd.to_datetime(
    data["DataDtTm"],
    format="%d%b%y:%H:%M:%S"
)

print("\nConverted timestamps:")
print(data.head())

#SORTING
data = data.sort_values("DataDtTm")

#Samppling frequency

data["time_diff"] = data["DataDtTm"].diff()

print("\nTime difference between readings:")
print(data["time_diff"].head(10))

#Glucose trend plot ( to show shameem sir, delete later)

plt.figure(figsize=(10,5))
plt.plot(data["DataDtTm"][:500], data["CGM"][:500])

plt.title("CGM Glucose trend(time-series) plot")
plt.xlabel("Time")
plt.ylabel("Glucose")

plt.show()
