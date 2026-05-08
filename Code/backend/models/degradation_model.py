import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.interpolate import PchipInterpolator

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

tracks = [
    "bahrain",
    "monza",
    "silverstone",
    "spa"
]

years = [2023, 2024, 2025]

compounds = ['SOFT', 'MEDIUM', 'HARD']

colors = {
    'SOFT': 'red',
    'MEDIUM': 'yellow',
    'HARD': 'gray',
    'INTERMEDIATE': 'green',
    'WET': 'blue'
}

models = {}

for track in tracks:

    print(f"\n\n========== TRAINING {track.upper()} ==========")

    dfs = []

    for year in years:

        file_path = os.path.join(
            BASE_DIR,
            "data",
            "processed",
            f"{track}_{year}.parquet"
        )

        if not os.path.exists(file_path):
            print(f"Missing file: {file_path}")
            continue

        temp_df = pd.read_parquet(file_path)

        temp_df['Season'] = year

        dfs.append(temp_df)

    if len(dfs) == 0:
        print(f"No data found for {track}")
        continue

    df = pd.concat(dfs, ignore_index=True)

    df = df[
        [
            'TyreLife',
            'FuelCorrectedLapTime',
            'Compound',
            'Driver',
            'Stint'
        ]
    ].copy()

    df = df.dropna()

    df = df[df['TyreLife'] <= 20]

    df = df[
        df['FuelCorrectedLapTime']
        < df['FuelCorrectedLapTime'].quantile(0.98)
    ]

    models[track] = {}

    plt.figure(figsize=(10, 6))

    for compound in compounds:

        print(f"\nTraining Model for compound: {compound}")

        compound_df = df[
            df['Compound'].str.upper() == compound
        ]

        grouped = compound_df.groupby(
            ['Driver', 'Stint']
        )

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

            group['Degradation'] = (
                group['FuelCorrectedLapTime'] - baseline
            )

            group = group[group['Degradation'] >= 0]

            group = group[group['Degradation'] < 3]

            threshold = group['Degradation'].quantile(0.95)

            group = group[
                group['Degradation'] <= threshold
            ]

            X_all.extend(group['TyreLife'].values)

            Y_all.extend(group['Degradation'].values)

        X = np.array(X_all, dtype=float)

        Y = np.array(Y_all, dtype=float)

        if len(X) == 0:
            print(f"No valid data for {compound}")
            continue

        plot_df = pd.DataFrame({
            'TyreLife': X,
            'Degradation': Y
        })

        avg_df = plot_df.groupby(
            'TyreLife'
        )['Degradation'].mean().reset_index()

        X_avg = avg_df['TyreLife'].values.astype(float)

        Y_avg = avg_df['Degradation'].values.astype(float)

        model = PchipInterpolator(
            X_avg,
            Y_avg
        )

        x_range = np.linspace(
            float(X_avg.min()),
            float(X_avg.max()),
            200
        )

        Y_smooth = model(x_range)

        models[track][compound] = model

        print(f"Model ready for {compound}")

        plt.scatter(
            X,
            Y,
            alpha=0.12,
            color=colors.get(compound, 'gray')
        )

        plt.plot(
            x_range,
            Y_smooth,
            linewidth=3,
            color=colors.get(compound, 'gray'),
            label=compound
        )

    plt.xlabel("TyreLife")

    plt.ylabel("Degradation")

    plt.title(
        f"{track.upper()} Tyre Degradation Comparison"
    )

    plt.legend()

    plt.grid(True)

    plt.show()

print(models.keys())