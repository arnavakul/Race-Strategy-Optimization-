import os
import pandas as pd

from api.models.weekend.practice_session_analyzer import (
    generate_practice_report
)


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(__file__)
        )
    )
)


PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

def extract_session_features(session_df):

    features = {}

    for driver in sorted(
        session_df["Driver"].unique()
    ):

        features[driver] = (
            generate_practice_report(
                session_df,
                driver
            )
        )

    return features


def build_prediction_dataset(track,year):
    
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

    fp1_df = pd.read_parquet(
        fp1_file
    )

    fp2_df = pd.read_parquet(
        fp2_file
    )

    fp3_df = pd.read_parquet(
        fp3_file
    )

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

    fp1_fastest = (
        fp1_df["LapTimeSeconds"]
        .min()
    )

    fp2_fastest = (
        fp2_df["LapTimeSeconds"]
        .min()
    )

    fp3_fastest = (
        fp3_df["LapTimeSeconds"]
        .min()
    )

    rows = []

    drivers = sorted(
        fp3_df["Driver"]
        .unique()
    )

    for driver in drivers:

        if (
            driver not in fp1_features
            or driver not in fp2_features
            or driver not in fp3_features
        ):
            continue

        team = (
            fp3_df[
                fp3_df["Driver"]
                == driver
            ]["Team"]
            .mode()
            .iloc[0]
        )

        row = {

            "driver":
                driver,

            "team":
                team,

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
                fp3_fastest
        }

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )

if __name__ == "__main__":

    TRACK = ["Abu Dhabi",
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
            "Suzuka",
            "Miami",
            "Shanghai",
            "Mexico City",
            "Las Vegas",
            "Baku",
            "Zandvoort"]
    
    YEAR = [2026]

    build_prediction_dataset(
        TRACK,
        YEAR
    )