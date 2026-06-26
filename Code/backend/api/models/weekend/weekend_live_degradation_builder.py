import pandas as pd
import numpy as np

from scipy.stats import linregress


def load_weekend_data(
    fp1_file,
    fp2_file,
    fp3_file
):

    fp1 = pd.read_parquet(
        fp1_file
    )

    fp2 = pd.read_parquet(
        fp2_file
    )

    fp3 = pd.read_parquet(
        fp3_file
    )

    return pd.concat(
        [
            fp1,
            fp2,
            fp3
        ],
        ignore_index=True
    )


def filter_long_run_laps(
    weekend_df
):

    return weekend_df[

        (weekend_df["IsAccurate"] == True)

        &

        (weekend_df["TrackStatus"] == "1")

        &

        (weekend_df["TyreLife"] >= 3)

        &

        (
            weekend_df["Compound"].isin(
                [
                    "SOFT",
                    "MEDIUM",
                    "HARD"
                ]
            )
        )

    ].copy()


def calculate_compound_degradation(
    weekend_df,
    compound
):

    compound_df = weekend_df[
        weekend_df["Compound"] == compound
    ]

    slopes = []

    grouped = compound_df.groupby(
        ["Driver", "Stint"],
        observed=True
    )

    for _, stint_df in grouped:

        if len(stint_df) < 5:
            continue

        if stint_df["TyreLife"].max() < 6:
            continue

        stint_df = stint_df.sort_values([
                "LapNumber",
                "TyreLife"
            ]
        )
        
        stint_df = stint_df.dropna(
            subset=[
                "TyreLife",
                "FuelCorrectedLapTime"
            ]
        )

        if len(stint_df) < 5:
            continue

        slope, _, _, _, _ = linregress(

            stint_df["TyreLife"],

            stint_df[
                "FuelCorrectedLapTime"
            ]
        )

        if pd.isna(slope):
            continue

        if slope < 0:
            continue

        if slope > 0.15:
            continue

        slopes.append(
            float(slope)
        )

    if len(slopes) == 0:

        return {
            "degradation": None,
            "stints": 0
        }

    if len(slopes) >= 8:

        confidence = "HIGH"

    elif len(slopes) >= 4:

        confidence = "MEDIUM"

    else:

        confidence = "LOW"

    return {

        "degradation":
            round(
                float(
                    np.median(slopes)
                ),
                4
            ),

        "stints":
            len(slopes),

        "confidence":
            confidence,

        "mean":
            round(
                float(
                    np.mean(slopes)
                ),
                4
            ),

        "min":
            round(
                float(
                    np.min(slopes)
                ),
                4
            ),

        "max":
            round(
                float(
                    np.max(slopes)
                ),
                4
            )
    }



def build_live_degradation_report(
    weekend_df
):

    report = {}

    compounds = [
        "SOFT",
        "MEDIUM",
        "HARD"
    ]

    for compound in compounds:

        report[compound] = (
            calculate_compound_degradation(
                weekend_df,
                compound
            )
        )

    return report

import json

from api.config.paths import REPORTS_PATH


def save_live_degradation_report(
    track,
    report
):

    output_file = (
        REPORTS_PATH
        / f"{track}_live_deg.json"
    )

    with open(
        output_file,
        "w"
    ) as f:

        json.dump(
            report,
            f,
            indent=4
        )

    return output_file

if __name__ == "__main__":

    weekend_df = load_weekend_data(

        r"data\processed\FP1\clean_laps\monaco_2026_FP1_clean.parquet",

        r"data\processed\FP2\clean_laps\monaco_2026_FP2_clean.parquet",

        r"data\processed\FP3\clean_laps\monaco_2026_FP3_clean.parquet"
    )

    weekend_df = filter_long_run_laps(
        weekend_df
    )

    report = build_live_degradation_report(
        weekend_df
    )

    print(
        "\n========== LIVE WEEKEND DEGRADATION ==========\n"
    )

    for compound, data in report.items():

        print(
            f"\n{compound}"
        )

        print(
            f"Degradation : {data['degradation']}"
        )

        print(
            f"Valid Stints: {data['stints']}"
        )

        print(
            f"Mean        : {data.get('mean')}"
        )

        print(
            f"Min         : {data.get('min')}"
        )

        print(
            f"Max         : {data.get('max')}"
        )
        
    save_live_degradation_report(
        "monaco_2026",
        report
    )