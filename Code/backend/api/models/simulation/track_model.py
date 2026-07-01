import pickle
import os
import copy

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "saved_models",
    "track_characteristics.pkl"
)

with open(MODEL_PATH, "rb") as f:
    TRACK_CHARACTERISTICS = pickle.load(f)

print(TRACK_CHARACTERISTICS.keys())


DEFAULT_TRACK = {

    "compound_pace_delta": {

        "SOFT": 0.0,
        "MEDIUM": 0.7,
        "HARD": 1.5,
        "INTERMEDIATE": 4.0,
        "WET": 7.0
    },

    # IMPORTANT
    "compound_deg": {

        "SOFT": 0.12,
        "MEDIUM": 0.08,
        "HARD": 0.05,
        "INTERMEDIATE": 0.09,
        "WET": 0.11
    },

    "cliff_age": {

        "SOFT": 12,
        "MEDIUM": 20,
        "HARD": 30,
        "INTERMEDIATE": 18,
        "WET": 15
    },

    "cliff_multiplier": {

        "SOFT": 0.08,
        "MEDIUM": 0.05,
        "HARD": 0.03,
        "INTERMEDIATE": 0.06,
        "WET": 0.09
    },

    "warmup_penalty": {

        "SOFT": 0.2,
        "MEDIUM": 0.5,
        "HARD": 0.8,
        "INTERMEDIATE": 0.7,
        "WET": 1.2
    }
}


def get_track_parameters(track):

    track = track.lower()

    # Exact lookup first
    if track in TRACK_CHARACTERISTICS:

        resolved_track = track

    else:

        # Find all matching seasons
        matches = [

            key

            for key in TRACK_CHARACTERISTICS.keys()

            if key.startswith(track + "_")

        ]

        if matches:

            # Pick newest season
            resolved_track = sorted(matches)[-1]

        else:

            resolved_track = None

    track_data = copy.deepcopy(DEFAULT_TRACK)

    if resolved_track:

        loaded = TRACK_CHARACTERISTICS[resolved_track]

        for key, value in loaded.items():

            if isinstance(value, dict):

                if key not in track_data:

                    track_data[key] = {}

                track_data[key].update(value)

            else:

                track_data[key] = value

    return track_data