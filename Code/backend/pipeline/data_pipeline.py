import fastf1
import pandas as pd
import os
import numpy as np
import gc

os.makedirs('./cache', exist_ok=True)
fastf1.Cache.enable_cache('./cache')

def process_race(year: int, track: str, save_path: str):
    session = fastf1.get_session(year, track, 'R')
    session.load()

    laps = session.laps[
        ['Driver', 'LapNumber', 'LapTime', 'Compound', 'TyreLife', 'Stint']
    ].copy()

    laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()
    laps.drop(columns=['LapTime'], inplace=True)

    laps = laps.dropna(subset=['LapTimeSeconds', 'TyreLife'])
    laps = laps[laps['TyreLife'] > 0]

    mean = laps['LapTimeSeconds'].mean()
    std = laps['LapTimeSeconds'].std()

    laps = laps[laps['LapTimeSeconds'] < (mean + 2 * std)].copy()

    k = 0.04  # seconds per lap loosing due to fuel usage. This value lies between 0.03s-0.05s. So we take the mean of the same. 
    laps['FuelCorrectedLapTime'] = (
        laps['LapTimeSeconds'] - k * laps['LapNumber']
    )

    fastest = laps['FuelCorrectedLapTime'].min()
    laps['NormalizedLapTime'] = laps['FuelCorrectedLapTime'] - fastest

    laps['DeltaLapTime'] = laps.groupby(
        ['Driver', 'Stint']
    )['FuelCorrectedLapTime'].transform(
        lambda x: x - x.iloc[0]
    )

    laps['Driver'] = laps['Driver'].astype('category')
    laps['Compound'] = laps['Compound'].astype('category')

    laps['LapNumber'] = laps['LapNumber'].astype('int16')
    laps['TyreLife'] = laps['TyreLife'].astype('int16')
    laps['Stint'] = laps['Stint'].astype('int8')

    laps['LapTimeSeconds'] = laps['LapTimeSeconds'].astype('float32')
    laps['FuelCorrectedLapTime'] = laps['FuelCorrectedLapTime'].astype('float32')
    laps['NormalizedLapTime'] = laps['NormalizedLapTime'].astype('float32')
    laps['DeltaLapTime'] = laps['DeltaLapTime'].astype('float32')

    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, f"{track.lower()}_{year}.parquet")

    laps.to_parquet(file_path, index=False)

    print(f"Saved → {file_path}")
    print(laps.head())

    del laps
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
    
    save_path = r"C:\DevProjects\Race Strategy Optimization\Code\backend\data\processed"
    
    os.makedirs(save_path, exist_ok=True)
    for track in tracks:
        for year in years:
            try:
                print(f"Starting {track} {year}")
                process_race(year, track, save_path)
            except Exception as e:
                print(f"Error processing {year}: {e}")

if __name__ == "__main__":
    run_pipeline()