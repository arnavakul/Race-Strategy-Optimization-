import fastf1
import pandas as pd
import os
import gc
import time

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

        if col in dataset.columns:

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


def process_session(
    year: int,
    track: str,
    session_type: str,
    save_path: str
):
    

    print(
        f"\nLoading "
        f"{track} "
        f"{year} "
        f"{session_type}..."
    )
    
    session_folder = os.path.join(
        save_path,
        session_type
    )

    clean_save_path = os.path.join(
        session_folder,
        "clean_laps"
    )

    clean_file = os.path.join(
        clean_save_path,
        f"{track.lower()}_{year}_{session_type}_clean.parquet"
    )

    if os.path.exists(clean_file):

        print(
            f"Skipping existing file: "
            f"{track} "
            f"{year} "
            f"{session_type}"
        )

        return
    

    session = fastf1.get_session(
        year,
        track,
        session_type
    )

    max_retries = 5

    for attempt in range(max_retries):

        try:

            session.load()
            
            print(
                f"Successfully loaded "
                f"{track} "
                f"{year} "
                f"{session_type}"
            )

            break

        except Exception as e:

            print(
                f"Retry {attempt+1}/{max_retries} "
                f"for {track} "
                f"{year} "
                f"{session_type}"
            )

            print(str(e))

            if (
                "500 calls/h" in str(e)
                or
                "RateLimitExceededError" in str(e)
            ):

                print(
                    "Rate limit reached."
                )

                print(
                    "Sleeping for 1 hour..."
                )

                time.sleep(600)

            else:

                time.sleep(15)

    else:

        print(
            f"Skipping "
            f"{track} "
            f"{year} "
            f"{session_type}"
        )

        return
    
    if session.laps.empty:

        print(
            f"No laps found for "
            f"{track} "
            f"{year} "
            f"{session_type}"
        )

        return
    
    full_laps = session.laps[
        [
            'Driver',
            'Team',
            'LapNumber',

            'LapTime',

            'Sector1Time',
            'Sector2Time',
            'Sector3Time',

            'Compound',
            'TyreLife',
            'FreshTyre',
            'Stint',

            'TrackStatus',
            'IsAccurate'
        ]
    ].copy()
    


    clean_laps = full_laps.copy()

    for dataset in [full_laps, clean_laps]:

        dataset['RaceYear'] = year

        dataset['Track'] = track

        dataset['SessionType'] = (
            session_type
        )

    full_laps = convert_time_columns(
        full_laps
    )

    clean_laps = convert_time_columns(
        clean_laps
    )

    clean_laps = clean_laps.dropna(
        subset=[
            'LapTimeSeconds',
            'TyreLife'
        ]
    )
    
    if clean_laps.empty:

        print(
            f"No valid laps after cleaning "
            f"{track} "
            f"{year} "
            f"{session_type}"
        )

        return

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
    
    if clean_laps.empty:

        print(
            f"No laps remaining after filtering "
            f"{track} "
            f"{year} "
            f"{session_type}"
        )

        return

    # ==================================
    # FUEL CORRECTION
    # ==================================

    k = 0.04

    clean_laps[
        'FuelCorrectedLapTime'
    ] = (

        clean_laps[
            'LapTimeSeconds'
        ]

        - k * clean_laps[
            'LapNumber'
        ]
    )

    # ==================================
    # NORMALIZATION
    # ==================================

    fastest = clean_laps[
        'FuelCorrectedLapTime'
    ].min()

    clean_laps[
        'NormalizedLapTime'
    ] = (

        clean_laps[
            'FuelCorrectedLapTime'
        ]

        - fastest
    )

    # ==================================
    # DELTA WITHIN STINT
    # ==================================

    clean_laps[
        'DeltaLapTime'
    ] = clean_laps.groupby(
        ['Driver', 'Stint']
    )[
        'FuelCorrectedLapTime'
    ].transform(
        lambda x: x - x.iloc[0]
    )

    # SESSION BEST LAP
    clean_laps[
        'SessionBestLap'
    ] = clean_laps.groupby(
        'Driver'
    )[
        'LapTimeSeconds'
    ].transform(
        'min'
    )

    # GAP TO SESSION BEST

    clean_laps[
        'GapToSessionBest'
    ] = (

        clean_laps[
            'LapTimeSeconds'
        ]

        -

        clean_laps[
            'SessionBestLap'
        ]
    )
    
    # print(full_laps.columns)

    full_laps = optimize_memory(
        full_laps
    )

    clean_laps = optimize_memory(
        clean_laps
    )

    session_folder = os.path.join(
        save_path,
        session_type
    )

    clean_save_path = os.path.join(
        session_folder,
        "clean_laps"
    )

    full_save_path = os.path.join(
        session_folder,
        "full_laps"
    )

    os.makedirs(
        clean_save_path,
        exist_ok=True
    )

    os.makedirs(
        full_save_path,
        exist_ok=True
    )

    clean_file = os.path.join(

        clean_save_path,

        f"{track.lower()}_"
        f"{year}_"
        f"{session_type}_"
        f"clean.parquet"
    )

    full_file = os.path.join(

        full_save_path,

        f"{track.lower()}_"
        f"{year}_"
        f"{session_type}_"
        f"full.parquet"
    )

    clean_laps.to_parquet(

        clean_file,

        index=False
    )

    full_laps.to_parquet(

        full_file,

        index=False
    )

    with open(
        "download_log.txt",
        "a",
        encoding="utf-8"
    ) as f:

        f.write(
            f"{track},"
            f"{year},"
            f"{session_type}\n"
        )

    print(
        f"\nSaved Clean Dataset → "
        f"{clean_file}"
    )

    print(
        f"Saved Full Dataset → "
        f"{full_file}"
    )

    print(
        "\nClean Dataset Shape:"
    )

    print(
        clean_laps.shape
    )

    print(
        "\nFull Dataset Shape:"
    )

    print(
        full_laps.shape
    )

    del full_laps

    del clean_laps

    gc.collect()


def run_weekend_pipeline():
    
    print("\nWEEKEND PIPELINE STARTED\n")

    # years = [2022, 2023, 2024, 2025, 2026]
    years = [2026]
    # # tracks = [
    #     "Abu Dhabi",
    #     "Austria",
    #     "Bahrain",
    #     "Barcelona",
    #     "Brazil",
    #     "COTA",
    #     "Hungary",
    #     "Jeddah",
    #     "Melbourne",
    #     "Monaco",
    #     "Monza",
    #     "Montreal",
    #     "Qatar",
    #     "Silverstone",
    #     "Singapore",
    #     "Spa",
    #     "Suzuka",
            # "Miami",
            # "Shanghai",
            # "Mexico City",
            # "Las Vegas",
            # "Imola",
            # "Baku",
            # "Zandvoort"
    # ]
    
    tracks = [
        "Austria",
    ]

    sessions = [
        "FP1",
        "FP2",
        "FP3",
        "Q",
        "R"
    ]

    save_path = (
        r"C:\DevProjects\Race Strategy Optimization"
        r"\Code\backend\data\processed"
    )
    
    print(
        f"\nSaving to: {save_path}"
    )

    for track in tracks:

        for year in years:

            for session_type in sessions:

                try:

                    print(
                        f"\n========================="
                    )

                    print(
                        f"Starting "
                        f"{track} "
                        f"{year} "
                        f"{session_type}"
                    )

                    print(
                        f"========================="
                    )

                    process_session(
                        year,
                        track,
                        session_type,
                        save_path
                    )
                    

                    time.sleep(2)

                except Exception as e:

                    print(
                        f"Error processing "
                        f"{track} "
                        f"{year} "
                        f"{session_type}: "
                        f"{e}"
                    )

if __name__ == "__main__":
    run_weekend_pipeline()