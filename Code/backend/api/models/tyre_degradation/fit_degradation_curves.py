import pandas as pd
import numpy as np
import os
import glob
import pickle

from scipy.optimize import curve_fit


def power_law(x, a, b):
    return a * (x ** b)


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)

processed_data_path = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

output_path = os.path.join(
    BASE_DIR,
    "models",
    "saved_models",
    "all_tracks_degradation.pkl"
)

parquet_files = glob.glob(
    os.path.join(processed_data_path, "*.parquet")
)

all_models = {}

for file in parquet_files:

    df = pd.read_parquet(file)

    track_name = os.path.basename(file).replace(".parquet", "")

    print(f"Processing: {track_name}")

    all_models[track_name] = {}

    for compound in ['SOFT', 'MEDIUM', 'HARD']:

        compound_df = df[df["Compound"] == compound]

        compound_df = compound_df.dropna(
            subset=["TyreLife", "LapTimeSeconds"]
        )

        if len(compound_df) < 5:
            continue

        x = compound_df["TyreLife"].values

        baseline = compound_df["LapTimeSeconds"].min()

        y = compound_df["LapTimeSeconds"] - baseline

        try:

            params, _ = curve_fit(
                power_law,
                x,
                y,
                maxfev=10000
            )

            a, b = params
            b = abs(b)

            all_models[track_name][compound] = (a, b)

            print(track_name, compound, a, b)

        except Exception as e:

            print(f"Failed for {track_name} {compound}")
            print(e)

with open(output_path, "wb") as f:
    pickle.dump(all_models, f)

print("\nMODEL SAVED\n")

with open(output_path, "rb") as f:

    test = pickle.load(f)

    print(test.keys())