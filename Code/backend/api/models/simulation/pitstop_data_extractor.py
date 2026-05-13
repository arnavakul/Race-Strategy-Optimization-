import os
import glob
import pickle
import pandas as pd

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(__file__)
        )
    )
)

processed_data_path = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "full_laps"
)

files = glob.glob(
    os.path.join(processed_data_path, "*.parquet")
)

if len(files) == 0:
    raise FileNotFoundError(
        f"No parquet files found in:\n{processed_data_path}"
    )

all_records = []

for file in files:

    print(f"\nProcessing: {file}")

    df = pd.read_parquet(file)

    pit_laps = df[
        df['PitInTime'].notna()
    ].copy()

    for _, lap in pit_laps.iterrows():

        driver = lap['Driver']

        lap_number = lap['LapNumber']

        driver_laps = df[
            df['Driver'] == driver
        ].copy()

        previous_laps = driver_laps[
            (
                driver_laps['LapNumber']
                >= lap_number - 3
            )
            &
            (
                driver_laps['LapNumber']
                < lap_number
            )
        ]

        clean_previous_laps = previous_laps[
            previous_laps['PitInTime'].isna()
        ]

        baseline = clean_previous_laps[
            'LapTimeSeconds'
        ].mean()

        if pd.isna(baseline):
            continue

        pit_in_lap_time = lap[
            'LapTimeSeconds'
        ]

        pit_out_lap = driver_laps[
            driver_laps['LapNumber']
            == lap_number + 1
        ]

        if len(pit_out_lap) == 0:
            continue

        pit_out_lap_time = pit_out_lap.iloc[0][
            'LapTimeSeconds'
        ]

        pit_loss = (
            (
                pit_in_lap_time
                +
                pit_out_lap_time
            )
            -
            (2 * baseline)
        )

        record = {
            'Driver': driver,
            'LapNumber': lap_number,
            'PitLoss': pit_loss,
            'Compound': lap['Compound'],
            'Track': lap['Track'],
            'RaceYear': lap['RaceYear']
        }

        all_records.append(record)


pitstop_df = pd.DataFrame(all_records)

pitstop_df = pitstop_df[
    pitstop_df['PitLoss'] > 10
]

pitstop_df = pitstop_df[
    pitstop_df['PitLoss'] < 45
]

print("\nPitstop Dataset:")
print(pitstop_df.head())

track_pit_loss = pitstop_df.groupby(
    'Track'
)['PitLoss'].median()

track_pit_loss_dict = (
    track_pit_loss.to_dict()
)

print("\nTrack Pit Loss Model:")
print(track_pit_loss_dict)

save_path = os.path.join(
    BASE_DIR,
    "models",
    "saved_models"
)

os.makedirs(save_path, exist_ok=True)

save_file = os.path.join(
    save_path,
    "pitstop_loss.pkl"
)

with open(save_file, "wb") as f:

    pickle.dump(
        track_pit_loss_dict,
        f
    )

print(f"\nSaved → {save_file}")

pitstop_df.to_parquet(
    os.path.join(
        save_path,
        "pitstop_dataset.parquet"
    ),
    index=False
)