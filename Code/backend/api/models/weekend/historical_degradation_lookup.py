import pickle
import numpy as np
import json

from api.config.weekend_config import *

from api.config.paths import (
    SAVED_MODELS_PATH, REPORTS_PATH
)

def load_degradation_models():

    model_path = (
        SAVED_MODELS_PATH
        / "all_tracks_degradation.pkl"
    )

    with open(model_path, "rb") as f:
        return pickle.load(f)

def power_law(
    tyre_life,
    a,
    b
):

    return a * np.power(
        tyre_life,
        b
    )

def estimate_cliff_age(
    compound
):

    if compound == "SOFT":
        return 12

    elif compound == "MEDIUM":
        return 20

    elif compound == "HARD":
        return 28

    return 20

def predict_compound_degradation(
    track,
    compound,
    tyre_life,
    models
):

    if track not in models:
        return None

    if compound not in models[track]:
        return None

    a, b = models[track][compound]

    return power_law(
        tyre_life,
        a,
        b
    )


def get_latest_track_key(
    track,
    models
):

    track = track.lower()

    matching_keys = [
        key
        for key in models.keys()
        if key.startswith(track)
    ]

    if not matching_keys:
        return None

    return sorted(matching_keys)[-1]

def build_weekend_degradation_report(
    track
):

    models = load_degradation_models()


    track_key = get_latest_track_key(
        track,
        models
    )

    if track_key is None:
        raise ValueError(
            f"No degradation model found for {track}"
        )

    compounds = [
        "SOFT",
        "MEDIUM",
        "HARD"
    ]

    report = {}
    print(track.lower())

    for compound in compounds:

        lap5 = predict_compound_degradation(
            track_key,
            compound,
            5,
            models
        )

        lap10 = predict_compound_degradation(
            track_key,
            compound,
            10,
            models
        )

        lap15 = predict_compound_degradation(
            track_key,
            compound,
            15,
            models
        )

        if (
            lap5 is None
            or lap10 is None
            or lap15 is None
        ):
            continue
        
        deg_rate = (
            lap15 - lap5
        ) / 10
        
        report[compound] = {

            "lap_5":
                round(float(lap5),3),

            "lap_10":
                round(float(lap10),3),

            "lap_15":
                round(float(lap15),3),

            "deg_rate":
                round(float(deg_rate),4),
            "cliff_age":
                estimate_cliff_age(
                    compound
                )
        }

    # print("\nSANITY CHECK\n")

    # for compound, data in report.items():

    #     print(
    #         compound,
    #         "Total degradation:",
    #         round(
    #             data["lap_15"]
    #             -
    #             data["lap_5"],
    #             3
    #         )
    #     )

    return report


def get_fastest_compound(
    degradation_report
):

    scores = {}

    for compound,data in degradation_report.items():

        scores[compound] = (
            degradation_report[compound][
                "deg_rate"
            ]
        )

    if not scores:
        return None

    return min(
        scores,
        key=scores.get
    )

def generate_weekend_degradation(
    track
):

    degradation_report = (
        build_weekend_degradation_report(
            track
        )
    )

    fastest_compound = (
        get_fastest_compound(
            degradation_report
        )
    )

    result = {

        "track":
            track,

        "source":
            "historical",

        "fastest_compound":
            fastest_compound,

        "degradation":
            degradation_report
    }

    save_file = (
        REPORTS_PATH
        / f"{track}_historical_deg.json"
    )

    with open(
        save_file,
        "w"
    ) as f:

        json.dump(
            result,
            f,
            indent=4
        )

    print(
        f"\nSaved historical degradation report:\n"
        f"{save_file}"
    )

    return result


if __name__ == "__main__":

    report = generate_weekend_degradation(
        "monaco"
    )

    print(
        "\n========== WEEKEND DEGRADATION ==========\n"
    )

    print(report)