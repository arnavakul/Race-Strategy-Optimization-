import itertools

from api.models.weekend.weekend_team_strength_builder import (
    build_session_report,
    build_team_metrics,
    calculate_team_strength_score
)
from api.models.weekend.historical_degradation_lookup import(
    generate_weekend_degradation as historical_degradation
)

from api.models.weekend.weekend_live_degradation_builder import (
    build_live_degradation_report
)

from api.models.simulation.track_model import (
    get_track_parameters
)

from api.models.optimization.strategy_optimizer import (
    strategy_optimizer 
)

from api.models.weekend.weekend_tyre_model import (
    build_weekend_tyre_model
)

from api.config.race_calendar import (
    RACE_CALENDAR
)

import pandas as pd
import json

from api.config.paths import REPORTS_PATH

def load_weekend_data(
    fp1_file,
    fp2_file,
    fp3_file
):

    fp1 = pd.read_parquet(fp1_file)

    fp2 = pd.read_parquet(fp2_file)

    fp3 = pd.read_parquet(fp3_file)

    session_df = pd.concat(
        [
            fp1,
            fp2,
            fp3
        ],
        ignore_index=True
    )

    return session_df

def build_race_context(
    track,
    weekend_df
):

    session_report = build_session_report(
        weekend_df
    )

    team_metrics = build_team_metrics(
        session_report
    )

    team_strength = calculate_team_strength_score(
        team_metrics
    )

    historical_deg = historical_degradation(
        track
    )

    live_deg = build_live_degradation_report(
        weekend_df
    )
    
    weekend_tyre_model = build_weekend_tyre_model(

        historical_deg,

        live_deg
    )
    
    print("\n========== WEEKEND TYRE MODEL ==========\n")

    for compound, data in weekend_tyre_model.items():

        print(compound)

        print(data)

        print()
    
    track_characteristics = get_track_parameters(track)

    total_laps = RACE_CALENDAR[track]["laps"]

    race_context = {

        "track": track,

        "total_laps": total_laps,

        "team_strength":
            team_strength.to_dict(
                orient="records"
            ),

        "historical_degradation":
            historical_deg,

        "live_degradation":
            live_deg,

        "weekend_tyre_model":
            weekend_tyre_model,

        "track_characteristics":
            track_characteristics
    }
    
    return race_context

def run_strategy_optimizer(
    race_context
):

    strategy = strategy_optimizer(

        race_context=race_context

    )

    return strategy

def save_strategy_report(
    strategy,
    track
):
    
    save_path = (
        REPORTS_PATH
        
        /f"{track}_weekend_strategy.json"
    )
    
    with open(
        save_path,
        "w"
    )as f:
        
        json.dump(
            strategy,
            f,
            indent=4,
            default=str
        )
        
    print(

        f"\nSaved Report:\n{save_path}"

    )

def generate_weekend_strategy(

    track,

    fp1_file,

    fp2_file,

    fp3_file

):
    
    weekend_df = load_weekend_data(
        fp1_file,
        fp2_file,
        fp3_file
    )
    
    race_context = build_race_context(
        track,
        weekend_df
    )
    
    print("\n========== RACE CONTEXT ==========\n")

    print(json.dumps(
        race_context,
        indent=4,
        default=str
    ))
    
    strategy = run_strategy_optimizer(
        race_context
    )
    
    result = {

        "race_context": race_context,

        "strategy": strategy

    }

    save_strategy_report(
        result,
        track
    )

    return result

if __name__ == "__main__":

    result = generate_weekend_strategy(

        track="barcelona",
        

        fp1_file=(
            r"data\processed\FP1\clean_laps"
            r"\barcelona_2026_FP1_clean.parquet"
        ),

        fp2_file=(
            r"data\processed\FP2\clean_laps"
            r"\barcelona_2026_FP2_clean.parquet"
        ),

        fp3_file=(
            r"data\processed\FP3\clean_laps"
            r"\barcelona_2026_FP3_clean.parquet"
        )

    )

    print("\n========== WEEKEND STRATEGY ==========\n")

    print(result)