import os

TRACKS = [
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
    "Suzuka",
    "Miami",
    "Shanghai",
    "Mexico City",
    "Las Vegas",
    "Baku",
    "Zandvoort"
]

TRAIN_YEARS = [
    2022,
    2023,
    2024,
    2025
]

PREDICTION_YEAR = 2026

DEFAULT_TRACK = "Monaco"

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

WEEKEND_DATASET_DIR = os.path.join(
    BASE_DIR,
    "data",
    "weekend_datasets"
)

QUALIFYING_DATASET_DIR = os.path.join(
    BASE_DIR,
    "data",
    "qualifying_datasets"
)

PREDICTION_DATASET_DIR = os.path.join(
    BASE_DIR,
    "data",
    "prediction_datasets"
)

PREDICTIONS_DIR = os.path.join(
    BASE_DIR,
    "data",
    "predictions"
)