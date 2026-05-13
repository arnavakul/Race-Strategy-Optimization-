import fastf1
import pandas as pd
import os
import gc

os.makedirs('./cache', exist_ok=True)
fastf1.Cache.enable_cache('./cache')


def convert_time_columns(dataset):

    dataset['LapTimeSeconds'] = (
        dataset['LapTime']
        .dt.total_seconds()
    )

    time_columns = [
        'Sector1Time',
        'Sector2Time',
        'Sector3Time'
    ]

    for col in time_columns:

        dataset[f'{col}Seconds'] = (
            dataset[col]
            .dt.total_seconds()
        )

    dataset.drop(
        columns=[
            'LapTime',
            'Sector1Time',
            'Sector2Time',
            'Sector3Time'
        ],
        inplace=True
    )

    return dataset


def optimize_memory(dataset):

    categorical_cols = [
        'Driver',
        'Team',
        'Compound',
        'Track'
    ]

    for col in categorical_cols:
        dataset[col] = dataset[col].astype('category')

    int_cols = {
        'LapNumber': 'int16',
        'TyreLife': 'int16',
        'Stint': 'int8',
        'Position': 'int8',
        'RaceYear': 'int16'
    }

    for col, dtype in int_cols.items():
        dataset[col] = dataset[col].astype(dtype)

    float_cols = [
        'LapTimeSeconds',
        'Sector1TimeSeconds',
        'Sector2TimeSeconds',
        'Sector3TimeSeconds'
    ]

    for col in float_cols:

        if col in dataset.columns:
            dataset[col] = dataset[col].astype('float32')

    extra_float_cols = [
        'FuelCorrectedLapTime',
        'NormalizedLapTime',
        'DeltaLapTime'
    ]

    for col in extra_float_cols:

        if col in dataset.columns:
            dataset[col] = dataset[col].astype('float32')

    return dataset


def process_race(year: int, track: str, save_path: str):

    print(f"\nLoading {track} {year}...")

    session = fastf1.get_session(year, track, 'R')
    session.load()

    full_laps = session.laps[
        [
            'Driver',
            'Team',
            'LapNumber',
            'Position',

            'LapTime',

            'Sector1Time',
            'Sector2Time',
            'Sector3Time',

            'Compound',
            'TyreLife',
            'FreshTyre',
            'Stint',

            'PitInTime',
            'PitOutTime',

            'TrackStatus',
            'IsAccurate'
        ]
    ].copy()

    clean_laps = full_laps.copy()

    for dataset in [full_laps, clean_laps]:

        dataset['RaceYear'] = year
        dataset['Track'] = track

    full_laps = convert_time_columns(full_laps)
    clean_laps = convert_time_columns(clean_laps)

    clean_laps = clean_laps.dropna(
        subset=[
            'LapTimeSeconds',
            'TyreLife'
        ]
    )

    clean_laps = clean_laps[
        clean_laps['IsAccurate'] == True
    ].copy()

    clean_laps = clean_laps[
        clean_laps['TrackStatus'] == '1'
    ].copy()

    clean_laps = clean_laps[
        clean_laps['TyreLife'] > 0
    ].copy()

    q99 = clean_laps[
        'LapTimeSeconds'
    ].quantile(0.99)

    clean_laps = clean_laps[
        clean_laps['LapTimeSeconds'] < q99
    ].copy()

    k = 0.04

    clean_laps['FuelCorrectedLapTime'] = (
        clean_laps['LapTimeSeconds']
        - k * clean_laps['LapNumber']
    )

    fastest = clean_laps[
        'FuelCorrectedLapTime'
    ].min()

    clean_laps['NormalizedLapTime'] = (
        clean_laps['FuelCorrectedLapTime']
        - fastest
    )

    clean_laps['DeltaLapTime'] = clean_laps.groupby(
        ['Driver', 'Stint']
    )['FuelCorrectedLapTime'].transform(
        lambda x: x - x.iloc[0]
    )

    full_laps = optimize_memory(full_laps)
    clean_laps = optimize_memory(clean_laps)

    clean_save_path = os.path.join(
        save_path,
        "clean_laps"
    )

    full_save_path = os.path.join(
        save_path,
        "full_laps"
    )

    os.makedirs(clean_save_path, exist_ok=True)
    os.makedirs(full_save_path, exist_ok=True)

    clean_file = os.path.join(
        clean_save_path,
        f"{track.lower()}_{year}_clean.parquet"
    )

    full_file = os.path.join(
        full_save_path,
        f"{track.lower()}_{year}_full.parquet"
    )

    clean_laps.to_parquet(
        clean_file,
        index=False
    )

    full_laps.to_parquet(
        full_file,
        index=False
    )

    print(f"\nSaved Clean Dataset → {clean_file}")
    print(f"Saved Full Dataset → {full_file}")

    print("\nClean Dataset Shape:")
    print(clean_laps.shape)

    print("\nFull Dataset Shape:")
    print(full_laps.shape)

    del full_laps
    del clean_laps

    gc.collect()


def run_pipeline():

    years = [2022, 2023, 2024, 2025]

    tracks = [
        "Abu Dhabi",
        "Austria",
        "Bahrain",
        "Barcelona",
        "Brazil",
        "COTA",
        "Hungary",
        "Jeddah",
        "Melbourne",
        "Monaco",
        "Monza",
        "Montreal",
        "Qatar",
        "Silverstone",
        "Singapore",
        "Spa",
        "Suzuka"
    ]

    save_path = (
        r"C:\DevProjects\Race Strategy Optimization"
        r"\Code\backend\data\processed"
    )

    os.makedirs(save_path, exist_ok=True)

    for track in tracks:

        for year in years:

            try:

                print(f"\n=========================")
                print(f"Starting {track} {year}")
                print(f"=========================")

                process_race(
                    year,
                    track,
                    save_path
                )

            except Exception as e:

                print(
                    f"Error processing {track} {year}: {e}"
                )


if __name__ == "__main__":
    run_pipeline()