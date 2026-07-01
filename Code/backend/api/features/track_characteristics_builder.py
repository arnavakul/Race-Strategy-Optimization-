import pickle
import numpy as np
import pandas as pd

from pathlib import Path


# Paths

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
    .parent
)

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "full_laps"
)

SAVE_PATH = (
    BASE_DIR
    / "api"
    / "models"
    / "saved_models"
    / "track_characteristics.pkl"
)

print("\nBASE_DIR:")
print(BASE_DIR)

print("\nDATA_PATH:")
print(DATA_PATH)

print("\nPATH EXISTS:")
print(DATA_PATH.exists())


# Load parquet files

files = list(
    DATA_PATH.glob("*.parquet")
)

print("\nFILES FOUND:")
print(files)

if len(files) == 0:

    raise ValueError(
        f"No parquet files found in: {DATA_PATH}"
    )

df_list = []

track_file_map = {}

for file in files:

    print(f"\nLoading: {file.name}")

    df = pd.read_parquet(file)

    file_name = file.stem.lower()

    track_key = (
        file_name
        .replace("_full", "")
        .replace(" ", "_")
    )

    df["TrackKey"] = track_key

    track_file_map[track_key] = file.name

    df_list.append(df)

all_laps = pd.concat(
    df_list,
    ignore_index=True
)

print(f"\nLoaded {len(all_laps)} laps")


# Clean laps

race_laps = all_laps.copy()

race_laps = race_laps.dropna(
    subset=[
        "Compound",
        "TyreLife",
        "LapTimeSeconds"
    ]
)

if "IsAccurate" in race_laps.columns:

    race_laps = race_laps[
        race_laps["IsAccurate"] == True
    ]


# Build characteristics

track_characteristics = {}

tracks = race_laps["TrackKey"].unique()

print("\nTRACKS FOUND:")
print(tracks)

for track in tracks:

    print(f"\nProcessing {track}")

    track_df = race_laps[
        race_laps["TrackKey"] == track
    ]

    compound_pace_delta = {}

    compound_deg = {}

    cliff_age = {}

    warmup_penalty = {}

    compound_reference_times = {}

    # Pace delta

    for compound in [
        "SOFT",
        "MEDIUM",
        "HARD"
    ]:

        compound_df = track_df[
            track_df["Compound"] == compound
        ]

        if len(compound_df) < 10:
            continue

        fresh_laps = compound_df[
            compound_df["TyreLife"] <= 3
        ]

        if len(fresh_laps) == 0:
            continue

        fresh_avg = fresh_laps[
            "LapTimeSeconds"
        ].mean()

        compound_reference_times[
            compound
        ] = fresh_avg

    if len(compound_reference_times) == 0:
        continue

    if "SOFT" in compound_reference_times:

        soft_reference = (
            compound_reference_times["SOFT"]
        )

    else:

        soft_reference = min(
            compound_reference_times.values()
        )

    for compound in compound_reference_times:

        delta = (
            compound_reference_times[compound]
            - soft_reference
        )

        compound_pace_delta[
            compound
        ] = round(float(delta), 3)

    # Deg + cliff + warmup

    for compound in [
        "SOFT",
        "MEDIUM",
        "HARD"
    ]:

        compound_df = track_df[
            track_df["Compound"] == compound
        ].copy()

        if len(compound_df) < 10:
            continue

        compound_df = compound_df.sort_values(
            by="TyreLife"
        )

        tyre_life = compound_df[
            "TyreLife"
        ].values

        lap_times = compound_df[
            "LapTimeSeconds"
        ].values

        if len(tyre_life) < 5:
            continue

        # Degradation fit

        slope, intercept = np.polyfit(
            tyre_life,
            lap_times,
            1
        )

        compound_deg[
            compound
        ] = round(float(slope), 4)

        # Cliff detection

        degradation_steps = np.diff(
            lap_times
        )

        cliff_detected = False

        for i, delta in enumerate(
            degradation_steps
        ):

            if delta > (
                compound_deg[compound] * 2.5
            ):

                cliff_age[
                    compound
                ] = int(
                    tyre_life[i]
                )

                cliff_detected = True

                break

        if not cliff_detected:

            default_cliff = {
                "SOFT": 15,
                "MEDIUM": 22,
                "HARD": 30
            }

            cliff_age[
                compound
            ] = default_cliff[compound]

        # Warmup penalty

        outlap_df = compound_df[
            compound_df["TyreLife"] <= 2
        ]

        stable_df = compound_df[
            (
                compound_df["TyreLife"] >= 4
            )
            &
            (
                compound_df["TyreLife"] <= 8
            )
        ]

        if (
            len(outlap_df) > 0
            and
            len(stable_df) > 0
        ):

            outlap_avg = outlap_df[
                "LapTimeSeconds"
            ].mean()

            stable_avg = stable_df[
                "LapTimeSeconds"
            ].mean()

            penalty = (
                outlap_avg
                - stable_avg
            )

            penalty = max(
                0.0,
                penalty
            )

            warmup_penalty[
                compound
            ] = round(
                float(penalty),
                3
            )

        else:

            default_warmup = {
                "SOFT": 0.5,
                "MEDIUM": 0.8,
                "HARD": 1.2
            }

            warmup_penalty[
                compound
            ] = default_warmup[
                compound
            ]
            
    degradation_samples = {}

    for compound in [

        "SOFT",
        "MEDIUM",
        "HARD"

    ]:

        degradation_samples[
            compound
        ] = int(

            len(

                track_df[
                    (
                        track_df["Compound"] == compound
                    )

                    &

                    (
                        track_df["TyreLife"] >= 3
                    )
                ]

            )

        )

    compound_usage = {}

    for compound in [

        "SOFT",
        "MEDIUM",
        "HARD"

    ]:

        compound_usage[
            compound
        ] = int(

            len(

                track_df[
                    track_df["Compound"] == compound
                ]

            )

        )

    track_characteristics[
        track
    ] = {

        "total_laps":
            int(
                track_df["LapNumber"].max()
            ),

        "average_race_lap":
            round(
                float(
                    track_df[
                        "LapTimeSeconds"
                    ].mean()
                ),
                3
            ),
        
        "compound_usage":
        compound_usage,

        "compound_pace_delta":
            compound_pace_delta,

        "historical_degradation":
            compound_deg,

        "cliff_age":
            cliff_age,

        "warmup_penalty":
            warmup_penalty,
            
        "degradation_samples":
            degradation_samples,
    }


# Save model

with open(SAVE_PATH, "wb") as f:

    pickle.dump(
        track_characteristics,
        f
    )

print("\nSaved track characteristics.")

print("\nSAVE PATH:")
print(SAVE_PATH)

print("\nDone.")