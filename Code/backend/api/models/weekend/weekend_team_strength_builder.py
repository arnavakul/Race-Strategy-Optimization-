#Function responsibility - 
#Suppose FP2 contains:

# VER  Red Bull
# TSU  Red Bull

# NOR  McLaren
# PIA  McLaren

# creates this 

# driver	team	best_lap	long_run_pace
# VER	  Red Bull	 72.3	    73.8
# TSU	  Red Bull	 72.7	    74.0
# NOR	  McLaren	 72.1	    73.5


from api.models.weekend.practice_session_analyzer import (
    generate_practice_report
)

import pandas as pd


# -------------------------------------------------------
# DRIVER SESSION REPORT
# -------------------------------------------------------

def build_session_report(
    session_df
):

    reports = []

    drivers = sorted(
        session_df["Driver"].unique()
    )

    for driver in drivers:

        report = generate_practice_report(
            session_df,
            driver
        )

        team = (
            session_df[
                session_df["Driver"] == driver
            ]["Team"]
            .mode()
            .iloc[0]
        )

        report["team"] = team

        reports.append(report)

    return pd.DataFrame(reports)


# -------------------------------------------------------
# TEAM METRICS
# -------------------------------------------------------

def build_team_metrics(
    session_report
):

    team_df = (

        session_report

        .groupby(
            "team",
            as_index=False
        )

        .agg(

            {

                "best_lap":
                    "mean",

                "long_run_pace":
                    "mean",

                "tyre_management":
                    "mean",

                "consistency":
                    "mean"

            }

        )

    )

    return team_df


# -------------------------------------------------------
# NORMALIZATION
# -------------------------------------------------------

def normalize(series):

    denominator = (

        series.max()

        - series.min()

    )

    if denominator == 0:

        return pd.Series(

            50.0,

            index=series.index

        )

    return (

        (

            series.max()

            - series

        )

        / denominator

    ) * 100


# -------------------------------------------------------
# TEAM STRENGTH
# -------------------------------------------------------

def calculate_team_strength_score(
    team_metrics
):

    df = team_metrics.copy()

    df["quali_score"] = normalize(

        df["best_lap"]

    )

    df["race_score"] = normalize(

        df["long_run_pace"]

    )

    df["tyre_management_score"] = normalize(

        df["tyre_management"]

    )

    df["consistency_score"] = normalize(

        df["consistency"]

    )

    df["team_strength"] = (

        df["quali_score"] * 0.25

        +

        df["race_score"] * 0.45

        +

        df["tyre_management_score"] * 0.20

        +

        df["consistency_score"] * 0.10

    )

    df = (

        df

        .sort_values(

            "team_strength",

            ascending=False

        )

        .reset_index(

            drop=True

        )

    )

    return df


# -------------------------------------------------------
# TESTING
# -------------------------------------------------------

if __name__ == "__main__":

    fp2_file = (

        r"C:\DevProjects\Race Strategy Optimization"

        r"\Code\backend\data\processed"

        r"\FP2\clean_laps"

        r"\monaco_2026_FP2_clean.parquet"

    )

    fp2_df = pd.read_parquet(

        fp2_file

    )

    print(

        "\n========== DRIVER REPORT ==========\n"

    )

    session_report = build_session_report(

        fp2_df

    )

    print(

        session_report.head(20)

    )

    print(

        "\n========== TEAM METRICS ==========\n"

    )

    team_metrics = build_team_metrics(

        session_report

    )

    print(team_metrics)

    strength_df = calculate_team_strength_score(

        team_metrics

    )

    print(

        "\n========== WEEKEND TEAM STRENGTH ==========\n"

    )

    print(

        strength_df[

            [

                "team",

                "quali_score",

                "race_score",

                "tyre_management_score",

                "consistency_score",

                "team_strength"

            ]

        ]

    )

    print(

        "\nDrivers:",

        len(session_report)

    )

    print(

        "Teams:",

        len(team_metrics)

    )