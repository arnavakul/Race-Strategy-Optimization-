import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.isotonic import IsotonicRegression

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
file_path = os.path.join(BASE_DIR, "data", "processed", "bahrain_2023.parquet")

df = pd.read_parquet(file_path)

df = df[['TyreLife','FuelCorrectedLapTime','Compound','Driver','Stint']].copy()
df = df.dropna()
df = df[df['TyreLife'] <= 20]
df = df[df['FuelCorrectedLapTime'] < df['FuelCorrectedLapTime'].quantile(0.98)]

soft_df = df[df['Compound'].str.upper() == 'SOFT']

grouped = soft_df.groupby(['Driver', 'Stint'])

X_all = []
Y_all = []

for _, group in grouped:
    group = group.sort_values('TyreLife')

    if len(group) < 6:
        continue

    group = group[group['TyreLife'] >= 3]

    if len(group) < 4:
        continue

    baseline = group['FuelCorrectedLapTime'].min()
    group = group.copy()
    group['Degradation'] = group['FuelCorrectedLapTime'] - baseline

    group = group[group['Degradation'] >= 0]
    group = group[group['Degradation'] < 3]

    threshold = group['Degradation'].quantile(0.95)
    group = group[group['Degradation'] <= threshold]

    X_all.extend(group['TyreLife'].values)
    Y_all.extend(group['Degradation'].values)

X = np.array(X_all, dtype=float)
Y = np.array(Y_all, dtype=float)

if len(X) == 0:
    raise ValueError("No valid data points")

sort_idx = np.argsort(X)
X_sorted = X[sort_idx]
Y_sorted = Y[sort_idx]  

iso = IsotonicRegression(increasing=True)
Y_iso = iso.fit_transform(X_sorted, Y_sorted)

print("Model ready (isotonic regression)")

plt.scatter(X, Y, alpha=0.3)
plt.plot(X_sorted, Y_iso, linewidth=2)
plt.xlabel("TyreLife")
plt.ylabel("Degradation")
plt.title("Tyre Degradation Curve - SOFT (Monotonic Model)")
plt.show()