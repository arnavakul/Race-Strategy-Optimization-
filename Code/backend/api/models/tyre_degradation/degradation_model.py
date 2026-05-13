import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle

from scipy.optimize import curve_fit

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

tracks = [
    "abu_dhabi",
    "austria",
    "bahrain",
    "barcelona",
    "brazil",
    "cota",
    "hungary",
    "jeddah",
    "melbourne",
    "monaco",
    "monza",
    "montreal",
    "qatar",
    "silverstone",
    "singapore",
    "spa",
    "suzuka"
]

years = [2022, 2023, 2024, 2025]

compounds = ["SOFT", "MEDIUM", "HARD"]

colors = {
    "SOFT": "red",
    "MEDIUM": "yellow",
    "HARD": "gray"
}

compound_max_life = {
    "SOFT": 15,
    "MEDIUM": 20,
    "HARD": 25
}

min_stint_lengths = {
    "SOFT": 4,
    "MEDIUM": 6,
    "HARD": 6
}

min_samples = {
    "SOFT": 6,
    "MEDIUM": 8,
    "HARD": 8
}


def power_law(x, a, b):
    return a * np.power(x, b)


models = {}

for track in tracks:

    print(f"\n========== TRAINING {track.upper()} ==========")

    dfs = []

    for year in years:

        file_path = os.path.join(
            BASE_DIR,
            "data",
            "processed",
            "clean_laps",
            f"{track}_{year}.parquet"
        )

        if not os.path.exists(file_path):
            print(f"Missing file: {file_path}")
            continue

        temp_df = pd.read_parquet(file_path)
        temp_df["Season"] = year
        dfs.append(temp_df)

    if len(dfs) == 0:
        print(f"No data found for {track}")
        continue

    df = pd.concat(dfs, ignore_index=True)

    required_columns = [
        "TyreLife",
        "FuelCorrectedLapTime",
        "Compound",
        "Driver",
        "Stint"
    ]

    missing_cols = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_cols:
        print(f"Missing columns in {track}: {missing_cols}")
        continue

    df = df[required_columns].copy()

    df["Compound"] = (
        df["Compound"]
        .astype(str)
        .str.upper()
    )

    df["TyreLife"] = pd.to_numeric(
        df["TyreLife"],
        errors="coerce"
    )

    df["FuelCorrectedLapTime"] = pd.to_numeric(
        df["FuelCorrectedLapTime"],
        errors="coerce"
    )

    df = df.dropna()

    df = df[df["TyreLife"] >= 2]
    df = df[df["TyreLife"] <= 25]

    df = df[
        df["FuelCorrectedLapTime"]
        < df["FuelCorrectedLapTime"].quantile(0.98)
    ]

    if len(df) == 0:
        print(f"No usable data after cleaning for {track}")
        continue

    models[track] = {}

    plt.figure(figsize=(12, 7))

    valid_compound_found = False

    for compound in compounds:

        print(f"\nTraining model for {compound}")

        compound_df = df[
            df["Compound"] == compound
        ].copy()

        if len(compound_df) < 20:
            print(f"Not enough raw data for {compound}")
            continue

        grouped = compound_df.groupby(["Driver", "Stint"])

        X_all = []
        Y_all = []

        for _, group in grouped:

            group = group.sort_values("TyreLife")

            group = group[
                group["TyreLife"]
                <= compound_max_life[compound]
            ]

            if len(group) < min_stint_lengths[compound]:
                continue

            baseline = (
                group
                .nsmallest(1, "FuelCorrectedLapTime")
                ["FuelCorrectedLapTime"]
                .iloc[0]
            )

            group = group.copy()

            group["Degradation"] = (
                group["FuelCorrectedLapTime"] - baseline
            )

            group = group[
                (group["Degradation"] >= 0) &
                (group["Degradation"] <= 3)
            ]

            if len(group) < min_stint_lengths[compound]:
                continue

            X_all.extend(group["TyreLife"].values.tolist())
            Y_all.extend(group["Degradation"].values.tolist())

        if len(X_all) == 0:
            print(f"No valid stint data for {compound}")
            continue

        X = np.array(X_all, dtype=float)
        Y = np.array(Y_all, dtype=float)

        plot_df = pd.DataFrame({
            "TyreLife": X,
            "Degradation": Y
        })

        avg_df = (
            plot_df
            .groupby("TyreLife")
            .agg(
                Degradation=("Degradation", "median"),
                Count=("Degradation", "size")
            )
            .reset_index()
        )

        avg_df = avg_df[
            avg_df["Count"] >= min_samples[compound]
        ]

        if len(avg_df) < 4:
            print(f"Not enough averaged points for {compound}")
            continue

        avg_df = avg_df.sort_values("TyreLife")

        X_avg = avg_df["TyreLife"].values.astype(float)
        Y_avg = avg_df["Degradation"].values.astype(float)

        try:

            params, _ = curve_fit(
                power_law,
                X_avg,
                Y_avg,
                p0=[0.1, 1.0],
                bounds=([0.005, 0.05], [5.0, 3.0]),
                maxfev=10000
            )

            a_fit, b_fit = params

        except Exception as e:

            print(f"Curve fitting failed for {compound}: {e}")
            continue

        x_range = np.linspace(
            float(X_avg.min()),
            float(X_avg.max()),
            300
        )

        y_smooth = power_law(
            x_range,
            a_fit,
            b_fit
        )

        models[track][compound] = {
            "a": float(a_fit),
            "b": float(b_fit)
        }

        valid_compound_found = True

        print(
            f"Model ready for {compound} | "
            f"a={a_fit:.4f}, b={b_fit:.4f} | "
            f"Lap5={power_law(5, a_fit, b_fit):.3f}s | "
            f"Lap10={power_law(10, a_fit, b_fit):.3f}s | "
            f"Lap15={power_law(15, a_fit, b_fit):.3f}s"
        )

        plt.scatter(
            X,
            Y,
            alpha=0.12,
            s=35,
            color=colors.get(compound, "gray")
        )

        plt.plot(
            x_range,
            y_smooth,
            linewidth=4,
            color=colors.get(compound, "gray"),
            label=compound
        )

    if not valid_compound_found:
        print(f"No valid compounds for {track}")
        plt.close()
        continue

    plt.xlabel("TyreLife", fontsize=16)
    plt.ylabel("Degradation (s)", fontsize=16)

    plt.title(
        f"{track.upper()} Tyre Degradation Comparison",
        fontsize=22
    )

    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.4)

    plt.xlim(1, 25)
    plt.ylim(-0.5, 3)

    output_dir = os.path.join(
        BASE_DIR,
        "outputs",
        "degradation_graphs"
    )

    os.makedirs(output_dir, exist_ok=True)

    save_path = os.path.join(
        output_dir,
        f"{track}_degradation.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"Saved graph: {save_path}")

    plt.close()

models_dir = os.path.join(BASE_DIR, "models")

os.makedirs(models_dir, exist_ok=True)

model_path = os.path.join(
    models_dir,
    "all_tracks_degradation.pkl"
)

with open(model_path, "wb") as f:
    pickle.dump(models, f)

print("\nAll models saved successfully")

print("\n========== MODEL SUMMARY ==========")

for track_name in models:

    for compound in models[track_name]:

        a = models[track_name][compound]["a"]
        b = models[track_name][compound]["b"]

        deg_5 = float(power_law(5, a, b))
        deg_10 = float(power_law(10, a, b))
        deg_15 = float(power_law(15, a, b))

        print(
            f"{track_name:15} | "
            f"{compound:6} | "
            f"a={a:.4f} "
            f"b={b:.3f} | "
            f"Lap5={deg_5:.3f}s "
            f"Lap10={deg_10:.3f}s "
            f"Lap15={deg_15:.3f}s"
        )

print("\nTracks trained:")
print(list(models.keys()))