import os
import pickle

from api.models.simulation.fuel_state import FuelState

from api.models.simulation.track_model import (
    get_track_parameters
)

from api.models.simulation.track_evolution_model import (
    get_track_grip
)

from api.models.simulation.driver_behavior_model import (
    DRIVER_PROFILES
)

from api.models.simulation.tyre_set_model import (
    get_freshness_penalty, get_performance_offset
)

from api.models.optimization.stochastic_models import(
    StochasticModels
)

from api.models.simulation.ml_pace_adapter import (
    get_ml_pace_adjustment
)

# Paths

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)

base_pace_path = os.path.join(
    BASE_DIR,
    "models",
    "saved_models",
    "track_base_pace.pkl"
)

deg_model_path = os.path.join(
    BASE_DIR,
    "models",
    "saved_models",
    "all_tracks_degradation.pkl"
)

# Load models

with open(base_pace_path, "rb") as f:

    track_base_pace = pickle.load(f)
    print(track_base_pace.keys())

with open(deg_model_path, "rb") as f:

    degradation_models = pickle.load(f)

# Base pace

def get_base_pace(track):

    if track in track_base_pace:
        return track_base_pace[track] + 4.5

    candidates = [

        key

        for key in track_base_pace

        if key.startswith(track.lower())

    ]

    if len(candidates) == 0:

        raise ValueError(
            f"No base pace model found for {track}"
        )

    latest = sorted(candidates)[-1]

    return track_base_pace[latest] + 4.5

# Tyre degradation

def get_degradation(
    track,
    compound,
    tyre_age,
    weekend_tyre_model=None
):

    track_data = get_track_parameters(track)

    if (

        weekend_tyre_model is not None

        and

        compound in weekend_tyre_model

    ):

        deg = weekend_tyre_model[compound]["weekend"]

    else:

        deg = track_data["compound_deg"][compound]

    cliff_age = (
        track_data["cliff_age"][compound]
    )

    cliff_multiplier = (
        track_data["cliff_multiplier"][compound]
    )

    # Fresh tyre gain
    if tyre_age <= 3:

        degradation = (
            -0.05 * tyre_age
        )

    # Normal degradation phase
    elif tyre_age <= 10:

        base_phase = -0.15

        degradation = (

            base_phase

            + (tyre_age - 3)

            * abs(deg)

            * 0.30
        )

    # High degradation phase
    elif tyre_age <= cliff_age:

        base_phase = (

            -0.15

            + (7 * abs(deg) * 0.30)
        )

        degradation = (

            base_phase

            + (tyre_age - 10)

            * abs(deg)

            * 0.45
        )

    # Tyre cliff
    else:

        base_phase = (

            -0.15

            + (7 * abs(deg) * 0.30)

            + ((cliff_age - 10)

            * abs(deg)

            * 0.45)
        )

        degradation = (

            base_phase

            + (tyre_age - cliff_age)

            * cliff_multiplier
        )

    return degradation

# Lap time engine

def compute_lap_time(
    track,
    compound,
    tyre_age,
    fuel_correction,
    current_lap,
    total_laps,
    driver_profile = "BALANCED",
    tyre_set = None,
    driver="VER",
    team="Red Bull",
    position=1,
    stint=1,
    race_year=2024,
    weekend_tyre_model = None,
    driver_rating=None,
    team_strength=None
):

    track_data = get_track_parameters(track)
    
    # Driver behavior profile
    driver_data = DRIVER_PROFILES[
        driver_profile
    ]

    pace_gain = driver_data[
        "pace_gain"
    ]

    deg_multiplier = driver_data[
        "deg_multiplier"
    ]

    # Track evolution grip
    track_grip = get_track_grip(
        current_lap,
        total_laps
    )

    compound_pace_delta = (
        track_data["compound_pace_delta"]
    )

    base_pace = get_base_pace(track)

    degradation = (

        get_degradation(

            track,

            compound,

            tyre_age,

            weekend_tyre_model

        )

        * deg_multiplier

    )

    compound_offset = (
        compound_pace_delta[compound]
    )
    
    ml_adjustment = get_ml_pace_adjustment(
        driver=driver,
        team=team,
        track=track.split("_")[0],
        compound=compound,
        tyre_life=tyre_age,
        position=position,
        stint=stint,
        race_year=race_year
    )
    
    ml_adjustment = (
        ml_adjustment
        - 5.0
    ) * 0.05

    # Core lap time model
    lap_time = (

        base_pace

        + compound_offset

        + degradation

        - fuel_correction

        - pace_gain
        
        + ml_adjustment 
    )   
    
    # Tyre freshness penalty

    freshness_penalty = 0
    performance_offset = 0
    
    if False:

        freshness_penalty = (
            get_freshness_penalty(
                tyre_set
            )
        )
        
        performance_offset =(
            get_performance_offset(
                tyre_set
            )
        )

    lap_time += freshness_penalty
    lap_time += performance_offset

    # Apply track evolution
    lap_time = (
        lap_time / track_grip
    )

    # Optional micro-randomness
    lap_time += (
        StochasticModels.sample_driver_variation()
    )
    
    return {

        "lap_time": float(lap_time),

        "base_pace": float(base_pace),

        "compound_offset": float(compound_offset),

        "degradation": float(degradation),

        "fuel_correction": float(fuel_correction),

        "track_grip": float(track_grip),
        
        "freshness_penalty": (
            float(freshness_penalty)
        ),
        
        "performance_offset": (
            float(performance_offset)
        ),
        
        "ml_adjustment":
            float(ml_adjustment),
        
    }

# Testing

from api.models.simulation.tyre_set_model import (
    create_tyre_set
)

test_tyre = create_tyre_set(

    compound="MEDIUM",

    freshness=0.82,

    heat_cycles=2,

    used_laps=6
)

def main():

    fuel = FuelState(
        starting_fuel=100,
        fuel_burn_per_lap=1.8
    )

    for lap in range(1, 21):

        fuel_correction = (
            fuel.getFuelCorrection()
        )

        result = compute_lap_time(

            track="bahrain_2022",

            compound="MEDIUM",

            tyre_age=lap,

            fuel_correction=fuel_correction,

            current_lap=lap,

            total_laps=57,

            driver_profile="AGGRESSIVE",
            
            tyre_set=test_tyre,
            
            
        )
        

        print(

            f"Lap {lap:>2} | "

            f"Lap Time: {result['lap_time']:.3f} | "

            f"Grip: {result['track_grip']:.3f} | "

            f"Deg: {result['degradation']:.3f}"
            
            f"Freshness Penalty: "
            
            f"{result['freshness_penalty']:.3f}"
            
            f"\nTyre Performance Offset: "
            
            f"{test_tyre['performance_offset']:.3f}\n"
            
        )        

        fuel.burnFuel()

if __name__ == "__main__":

    main()