import pandas as pd
from api.models.weekend.practice_session_analyzer import (
    generate_practice_report
)
from api.config.weekend_config import *
import os

def load_weekend_data(fp1_file,fp2_file,fp3_file,quali_file):
    fp1 = pd.read_parquet(
        fp1_file
    )
    
    fp2 = pd.read_parquet(
        fp2_file
    )
        
    fp3 = pd.read_parquet(
        fp3_file
    )
    
    qualifying = pd.read_parquet(
        quali_file
    )
    
    return (
        fp1,
        fp2,
        fp3,
        qualifying
    )

def extract_session_features(session_df):
    
    features = {}
    
    drivers = sorted(
        session_df["Driver"].unique()
    )
    
    for driver in drivers:
        
        features[driver] = (
            generate_practice_report(
                session_df,driver
            )
        )
        

    return features


def build_weekend_dataset(
    fp1_df,
    fp2_df,
    fp3_df,
    quali_df,
    track
):
    fp1_features = (
        extract_session_features(
            fp1_df
        )
    )
    
    fp2_features = (
    extract_session_features(
            fp2_df
        )
    )

    fp3_features = (
        extract_session_features(
            fp3_df
        )
    )
    
    fp1_fastest = fp1_df[
        "LapTimeSeconds"
    ].min()

    fp2_fastest = fp2_df[
        "LapTimeSeconds"
    ].min()

    fp3_fastest = fp3_df[
        "LapTimeSeconds"
    ].min()
    
    rows = []
    
    drivers = sorted(
        quali_df["Driver"].unique()
    )
    
    for driver in drivers:
        
        quali_driver = quali_df[
            quali_df["Driver"] == driver
        ]
        
        driver_team = (
            quali_driver["Team"]
            .mode()
            .iloc[0]
        )
        
        if quali_driver.empty:
            continue
        
        target_quali_time = (
            quali_driver[
                "LapTimeSeconds"
            ].min()
        )
        
        if (
            driver not in fp1_features
            or driver not in fp2_features
            or driver not in fp3_features
        ):
            continue
        
        row = {

            "driver": driver,
            
            "track": track,
            
            "team": driver_team,

            "fp1_best_lap":
                fp1_features[driver]["best_lap"],

            "fp2_best_lap":
                fp2_features[driver]["best_lap"],

            "fp3_best_lap":
                fp3_features[driver]["best_lap"],
            
            "avg_best_lap":
                (
                    fp1_features[driver]["best_lap"]
                    +
                    fp2_features[driver]["best_lap"]
                    +
                    fp3_features[driver]["best_lap"]
                ) / 3,

            "fp1_long_run":
                fp1_features[driver]["long_run_pace"],

            "fp2_long_run":
                fp2_features[driver]["long_run_pace"],

            "fp3_long_run":
                fp3_features[driver]["long_run_pace"],
                
            "avg_long_run":
                (
                        fp1_features[driver]["long_run_pace"]
                        +
                        fp2_features[driver]["long_run_pace"]
                        +
                        fp3_features[driver]["long_run_pace"]
                 ) / 3,

            "fp1_deg":
                fp1_features[driver]["degradation_rate"],

            "fp2_deg":
                fp2_features[driver]["degradation_rate"],

            "fp3_deg":
                fp3_features[driver]["degradation_rate"],

            "avg_deg":
                (
                    fp1_features[driver]["degradation_rate"]
                    +
                    fp2_features[driver]["degradation_rate"]
                    +
                    fp3_features[driver]["degradation_rate"]
                ) / 3,
            
            "fp2_minus_fp1":
                fp2_features[driver]["best_lap"]
                -
                fp1_features[driver]["best_lap"],

            "fp3_minus_fp2":
                fp3_features[driver]["best_lap"]
                -
                fp2_features[driver]["best_lap"],

            "fp3_minus_fp1":
                fp3_features[driver]["best_lap"]
                -
                fp1_features[driver]["best_lap"],
            
            "fp1_gap":
                fp1_features[driver]["best_lap"]
                -
                fp1_fastest,

            "fp2_gap":
                fp2_features[driver]["best_lap"]
                -
                fp2_fastest,

            "fp3_gap":
                fp3_features[driver]["best_lap"]
                -
                fp3_fastest,

            "target_quali_time":
                target_quali_time
        }
        
        rows.append(row)
        
        
    weekend_dataset = pd.DataFrame(
        rows
    )
            
    return weekend_dataset


def build_track_history(
    tracks,
    years
):
    
    
    for track in tracks:
        
        for year in years:
            print(
                f"\nBuilding {track} {year}"
            )
            
            fp1_file = os.path.join(
                PROCESSED_DIR,
                "FP1",
                "clean_laps",
                f"{track.lower()}_{year}_FP1_clean.parquet"
            )
            
            fp2_file = os.path.join(
                PROCESSED_DIR,
                "FP2",
                "clean_laps",
                f"{track.lower()}_{year}_FP2_clean.parquet"
            )

            fp3_file = os.path.join(
                PROCESSED_DIR,
                "FP3",
                "clean_laps",
                f"{track.lower()}_{year}_FP3_clean.parquet"
            )

            quali_file = os.path.join(
                PROCESSED_DIR,
                "Q",
                "clean_laps",
                f"{track.lower()}_{year}_Q_clean.parquet"
            )
            
            try: 
                
                fp1_df,fp2_df,fp3_df,quali_df = (
                    load_weekend_data(
                        fp1_file,
                        fp2_file,
                        fp3_file,
                        quali_file
                    )
                )
                
                weekend_dataset = (
                    build_weekend_dataset(
                        fp1_df,
                        fp2_df,
                        fp3_df,
                        quali_df,
                        track
                    )
                )
                
                os.makedirs(
                    WEEKEND_DATASET_DIR,
                    exist_ok=True
                )

                save_file = os.path.join(
                    WEEKEND_DATASET_DIR,
                    f"{track.lower()}_{year}_weekend.parquet"
                )
                
                weekend_dataset.to_parquet(
                    save_file,
                    index=False
                )

                print(
                    f"Saved -> {save_file}"
                )

            except Exception as e:
                
                if not (
                    os.path.exists(fp1_file)
                    and os.path.exists(fp2_file)
                    and os.path.exists(fp3_file)
                    and os.path.exists(quali_file)
                ):
                    print(
                        f"Skipping {track} {year} "
                        f"(missing session files)"
                    )
                    continue
                print(e)

if __name__ == "__main__":

    build_track_history(
        TRACKS,
        TRAIN_YEARS
    )
