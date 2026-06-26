import os
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.model_selection import (
    train_test_split
)


def load_training_data():

    file_path = (
        r"C:\DevProjects\Race Strategy Optimization"
        r"\Code\backend\data\qualifying_datasets"
        r"\qualifying_training_dataset.parquet"
    )

    return pd.read_parquet(
        file_path
    )


def prepare_features(df):

    feature_columns = [

        "fp1_best_lap",
        "fp2_best_lap",
        "fp3_best_lap",

        "fp1_long_run",
        "fp2_long_run",
        "fp3_long_run",

        "fp1_deg",
        "fp2_deg",
        "fp3_deg",

        "avg_best_lap",
        "avg_long_run",
        "avg_deg",

        "fp1_gap",
        "fp2_gap",
        "fp3_gap",

        "fp2_minus_fp1",
        "fp3_minus_fp2",
        "fp3_minus_fp1",

        "driver_track_rating",

        "team_quali_strength",
        "team_race_strength",
        "team_combined_strength"
    ]

    X = df[
        feature_columns
    ]

    y = df[
        "target_quali_time"
    ]

    return X, y


def train_model(
    X_train,
    y_train
):

    model = RandomForestRegressor(

        n_estimators=200,

        max_depth=8,

        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    return model

def load_prediction_data(track,year):

    prediction_file = (
        f"{track.lower()}_{year}_prediction_dataset.parquet"
    )

    prediction_df = pd.read_parquet(
        prediction_file
    )

    ratings_df = pd.read_parquet(
        r"C:\DevProjects\Race Strategy Optimization"
        r"\Code\backend\data\driver_ratings"
        r"\driver_track_ratings.parquet"
    )

    ratings_df = ratings_df[
        ratings_df["Track"] == track
    ]

    team_strength_df = pd.read_parquet(
        r"C:\DevProjects\Race Strategy Optimization"
        r"\Code\backend\data\team_strength"
        r"\team_strength.parquet"
    )

    prediction_df = prediction_df.merge(
        ratings_df[
            [
                "Driver",
                "CombinedRating"
            ]
        ],
        left_on="driver",
        right_on="Driver",
        how="left"
    )

    prediction_df.rename(
        columns={
            "CombinedRating":
                "driver_track_rating"
        },
        inplace=True
    )

    prediction_df = prediction_df.merge(
        team_strength_df,
        left_on="team",
        right_on="Team",
        how="left"
    )

    prediction_df.rename(
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

    return prediction_df
def build_weekend_grid_prediction(
    prediction_df
):

    prediction_df = prediction_df.copy()

    prediction_df["fp3_rank_score"] = (
        1
        -
        prediction_df["fp3_best_lap"]
        .rank(ascending=True, pct=True)
    )

    prediction_df["fp2_rank_score"] = (
        1
        -
        prediction_df["fp2_best_lap"]
        .rank(ascending=True, pct=True)
    )

    prediction_df["fp1_rank_score"] = (
        1
        -
        prediction_df["fp1_best_lap"]
        .rank(ascending=True, pct=True)
    )

    prediction_df["improvement_score"] = (
        -prediction_df["fp3_minus_fp1"]
    )

    improvement_min = (
        prediction_df["improvement_score"]
        .min()
    )

    improvement_max = (
        prediction_df["improvement_score"]
        .max()
    )

    prediction_df["improvement_score"] = (
        (
            prediction_df["improvement_score"]
            -
            improvement_min
        )
        /
        (
            improvement_max
            -
            improvement_min
        )
    ) * 100

    prediction_df["QualifyingScore"] = (

        prediction_df["fp3_rank_score"] * 50

        +

        prediction_df["fp2_rank_score"] * 25

        +

        prediction_df["fp1_rank_score"] * 5

        +

        prediction_df["improvement_score"] * 0.10

        +

        prediction_df["driver_track_rating"] * 0.10

        +

        prediction_df["team_combined_strength"] * 0.10
    )

    prediction_df = (
        prediction_df
        .sort_values(
            "QualifyingScore",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    prediction_df[
        "PredictedGridPosition"
    ] = (
        prediction_df.index + 1
    )

    return prediction_df

def run_qualifying_prediction(
    track,
    year
):

    prediction_df = (
        load_prediction_data(
            track,
            year
        )
    )

    predicted_grid = (
        build_weekend_grid_prediction(
            prediction_df
        )
    )

    return predicted_grid

if __name__ == "__main__":

    training_df = (
        load_training_data()
    )

    X, y = (
        prepare_features(
            training_df
        )
    )

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42
    )

    model = train_model(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = (
        mean_squared_error(
            y_test,
            predictions
        ) ** 0.5
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    print(
        "\nMODEL PERFORMANCE\n"
    )

    print(
        f"MAE : {mae:.3f}"
    )

    print(
        f"RMSE: {rmse:.3f}"
    )

    print(
        f"R2  : {r2:.3f}"
    )

    importance_df = pd.DataFrame({

        "Feature":
            X.columns,

        "Importance":
            model.feature_importances_
    })

    importance_df = (
        importance_df
        .sort_values(
            "Importance",
            ascending=False
        )
    )

    print(
        "\nFEATURE IMPORTANCE\n"
    )

    print(
        importance_df
    )
    
    print(
    "\nMONACO 2026 QUALIFYING PREDICTION\n"
    )

    TRACK = "Monaco"
    YEAR = 2026

    prediction_df = (
        load_prediction_data(
            TRACK,
            YEAR
        )
    )

    predicted_grid = (
        build_weekend_grid_prediction(
            prediction_df
        )
    )

    fastest_fp3 = (
        predicted_grid[
            "fp3_best_lap"
        ].min()
    )

    predicted_grid[
        "PredictedLapTime"
    ] = (
        fastest_fp3
        +
        (
            predicted_grid[
                "QualifyingScore"
            ].max()
            -
            predicted_grid[
                "QualifyingScore"
            ]
        ) * 0.03
    )

    predicted_grid = (
        predicted_grid
        .sort_values(
            "PredictedLapTime"
        )
        .reset_index(
            drop=True
        )
    )

    predicted_grid[
        "PredictedGridPosition"
    ] = (
        predicted_grid.index + 1
    )

    print(
        predicted_grid[
            [
                "PredictedGridPosition",
                "driver",
                "team",
                "PredictedLapTime",
                "QualifyingScore"
            ]
        ]
    )

    save_folder = (
        r"C:\DevProjects\Race Strategy Optimization"
        r"\Code\backend\data\predictions"
    )

    os.makedirs(
        save_folder,
        exist_ok=True
    )

    predicted_grid[
        [
            "PredictedGridPosition",
            "driver",
            "team",
            "PredictedLapTime",
            "QualifyingScore"
        ]
    ].to_csv(
        os.path.join(
            save_folder,
            f"{TRACK.lower()}_{YEAR}_predicted_grid.csv"
        ),
        index=False
    )

    print(
        f"\nSaved -> "
        f"{TRACK.lower()}_{YEAR}_predicted_grid.csv"
    )