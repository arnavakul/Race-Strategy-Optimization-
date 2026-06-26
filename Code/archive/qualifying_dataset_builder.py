import os
import pandas as pd

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(__file__)
        )
    )
)

def load_weekend_data(track):

    folder = os.path.join(
        BASE_DIR,
        "data",
        "weekend_datasets",
        track
    )

    datasets = []

    years = [
        2022,
        2023,
        2024,
        2025
    ]

    for year in years:

        file_path = os.path.join(
            folder,
            f"{track.lower()}_{year}_weekend.parquet"
        )

        if not os.path.exists(file_path):

            print(
                f"Missing: {file_path}"
            )

            continue

        df = pd.read_parquet(
            file_path
        )

        df["year"] = year

        datasets.append(df)

    return pd.concat(
        datasets,
        ignore_index=True
    )

def load_driver_ratings():

    file_path = (
        r"C:\DevProjects\Race Strategy Optimization"
        r"\Code\backend\data\driver_ratings"
        r"\driver_track_ratings.parquet"
    )

    return pd.read_parquet(
        file_path
    )


def load_team_strength():

    file_path = (
        r"C:\DevProjects\Race Strategy Optimization"
        r"\Code\backend\data\team_strength"
        r"\team_strength.parquet"
    )

    return pd.read_parquet(
        file_path
    )


def build_qualifying_dataset(track):

    weekend_df = (
        load_weekend_data(track)
    )

    driver_ratings = (
        load_driver_ratings()
    )

    team_strength = (
        load_team_strength()
    )

    driver_ratings = (
        driver_ratings[
            driver_ratings["Track"]
            == track
        ]
    )

    merged_df = (
        weekend_df.merge(
            driver_ratings[
                [
                    "Driver",
                    "CombinedRating"
                ]
            ],
            left_on="driver",
            right_on="Driver",
            how="left"
        )
    )

    merged_df.rename(
        columns={
            "CombinedRating":
                "driver_track_rating"
        },
        inplace=True
    )

    merged_df = (
        merged_df.merge(
            team_strength,
            left_on="team",
            right_on="Team",
            how="left"
        )
    )

    merged_df.rename(
        columns={
            "QualiStrength":
                "team_quali_strength",

            "RaceStrength":
                "team_race_strength",

            "CombinedStrength":
                "team_combined_strength"
        },
        inplace=True
    )

    merged_df[
        "driver_track_rating"
    ] = merged_df[
        "driver_track_rating"
    ].fillna(
        merged_df[
            "driver_track_rating"
        ].mean()
    )

    merged_df[
        "team_quali_strength"
    ] = merged_df[
        "team_quali_strength"
    ].fillna(
        merged_df[
            "team_quali_strength"
        ].mean()
    )

    merged_df[
        "team_race_strength"
    ] = merged_df[
        "team_race_strength"
    ].fillna(
        merged_df[
            "team_race_strength"
        ].mean()
    )

    merged_df[
        "team_combined_strength"
    ] = merged_df[
        "team_combined_strength"
    ].fillna(
        merged_df[
            "team_combined_strength"
        ].mean()
    )

    return merged_df


if __name__ == "__main__":

    TRACK = "Monaco"

    qualifying_dataset = (
        build_qualifying_dataset(
            TRACK
        )
    )

    save_folder = os.path.join(
        BASE_DIR,
        "data",
        "qualifying_datasets"
    )

    os.makedirs(
        save_folder,
        exist_ok=True
    )

    parquet_file = os.path.join(
        save_folder,
        f"{TRACK.lower()}_qualifying_training_dataset.parquet"
    )

    csv_file = os.path.join(
        save_folder,
        f"{TRACK.lower()}_qualifying_training_dataset.csv"
    )

    qualifying_dataset.to_parquet(
        parquet_file,
        index=False
    )

    qualifying_dataset.to_csv(
        csv_file,
        index=False
    )

    print(
        f"\nDataset Shape: "
        f"{qualifying_dataset.shape}"
    )

    print(
        f"\nSaved:\n{parquet_file}"
    )