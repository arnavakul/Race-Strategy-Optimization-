from api.models.weekend.practice_session_analyzer import (
    generate_practice_report
)

import pandas as pd

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


def build_session_report(
    session_df
):
    
    reports = []
    
    drivers = sorted(
        session_df["Driver"].unique()
    )
    
    for driver in drivers:
        
        report = (
            generate_practice_report(
                session_df,
                driver
            )
        )
        
        team = (
            session_df[
                session_df["Driver"] == driver
            ]["Team"].mode().iloc[0]
        )
        
        report["team"] = team
        
        reports.append(
            report
        )
    return pd.DataFrame(
        reports
    )

def build_team_metrics(
    session_report
):
    
    team_df = (
        session_report.groupby(
            "team",
            as_index = False
        ).agg(
            {
                
            "best_lap":
                "mean",

            "long_run_pace":
                "mean",

            "degradation_rate":
                "mean",

            "consistency":
                "mean"
            }
        )
    )
    
    return team_df

def calculate_team_strength_score(
    team_metrics
):

    df = team_metrics.copy()

    df["quali_score"] = (

        (
            df["best_lap"].max()
            -
            df["best_lap"]
        )
        /
        (
            df["best_lap"].max()
            -
            df["best_lap"].min()
        )
    ) * 100

    df["race_score"] = (

        (
            df["long_run_pace"].max()
            -
            df["long_run_pace"]
        )
        /
        (
            df["long_run_pace"].max()
            -
            df["long_run_pace"].min()
        )
    ) * 100

    df["consistency_score"] = (

        (
            df["consistency"].max()
            -
            df["consistency"]
        )
        /
        (
            df["consistency"].max()
            -
            df["consistency"].min()
        )
    ) * 100

    df["team_strength"] = (

        df["quali_score"] * 0.30
        +
        df["race_score"] * 0.60
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


#TESTING CODE:
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

    session_report = (
        build_session_report(
            fp2_df
        )
    )

    print(
        session_report.head(20)
    )

    print(
        "\n========== TEAM METRICS ==========\n"
    )

    team_metrics = (
        build_team_metrics(
            session_report
        )
    )

    print(
        team_metrics
        .sort_values(
            "long_run_pace"
        )
        .reset_index(
            drop=True
        )
    )

    print(
        "\n========== SUMMARY ==========\n"
    )

    print(
        f"Drivers: {len(session_report)}"
    )

    print(
        f"Teams: {len(team_metrics)}"
    )
    
    strength_df = (
    calculate_team_strength_score(
        team_metrics
        )
    )

    print(
        "\n========== WEEKEND TEAM STRENGTH ==========\n"
    )

    print(
        strength_df[
            [
                "team",
                "team_strength"
            ]
        ]
    )