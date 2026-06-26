from api.config.weekend_config import *

from Code.archive.weekend_feature_builder import (
    build_track_history
)

from Code.archive.qualifying_dataset_builder import (
    build_qualifying_dataset
)

from Code.archive.qualifying_predictor import (
    train_model,run_qualifying_prediction
)

from Code.archive.weekend_prediction_builder import (
    build_prediction_dataset
)

def stage_1_build_historical_weekends():

    print(
        "\n========== STAGE 1 =========="
    )

    build_track_history(
        TRACKS,
        TRAIN_YEARS
    )


def stage_2_build_training_dataset():

    print(
        "\n========== STAGE 2 =========="
    )

    print(
        "Building qualifying training dataset..."
    )

def stage_3_train_model():

    print(
        "\n========== STAGE 3 =========="
    )

    print(
        "Training qualifying predictor..."
    )

def stage_4_build_prediction_dataset():

    print(
        "\n========== STAGE 4 =========="
    )

    print(
        "Building prediction dataset..."
    )

def stage_5_predict_grid():

    print(
        "\n========== STAGE 5 =========="
    )

    print(
        "Predicting qualifying grid..."
    )

def run_pipeline():

    stage_1_build_historical_weekends()

    stage_2_build_training_dataset()

    stage_3_train_model()

    stage_4_build_prediction_dataset()

    stage_5_predict_grid()


if __name__ == "__main__":

    run_pipeline()