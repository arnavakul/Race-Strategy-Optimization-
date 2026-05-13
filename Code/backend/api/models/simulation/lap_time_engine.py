import os
import pickle
import random
from fuel_state import FuelState

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

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
with open(base_pace_path, "rb") as f:
    track_base_pace = pickle.load(f)

with open(deg_model_path, "rb") as f:
    degradation_models = pickle.load(f)

# print(track_base_pace.keys())
# print(degradation_models.keys())

def get_base_pace(track):
    return track_base_pace[track]

def get_degradation(track,compound, tyre_age):
    
    compound_deg = {
        "SOFT": 0.08,
        "MEDIUM": 0.05,
        "HARD": 0.03
    }
    safe_age = max(1, tyre_age)

    base_deg = tyre_age * compound_deg[compound]
    noise = random.uniform(0, 0.01)
    
    deg =base_deg+noise
    
    return deg

print(get_degradation(
    "bahrain_2022",
    "MEDIUM",
    10
))

def compute_lap_time(
        track, 
        compound,
        tyre_age,
        fuel_correction
    ):
        base_pace = get_base_pace(track)
        
        
        degradation = get_degradation(
            track,compound,tyre_age
        )
                
        lap_time = base_pace + degradation - fuel_correction
        return {
            "lap_time": float(lap_time),
            "base_pace": float(base_pace),
            "degradation": float(degradation),
            "fuel_correction": float(fuel_correction)
        }
        
def main():

    fuel = FuelState(
        starting_fuel=100,
        fuel_burn_per_lap=1.8,
        fuel_effect_per_kg=0.035
    )

    fuel_correction = fuel.getFuelCorrection()

    result = compute_lap_time(
        track="bahrain_2022",
        compound="MEDIUM",
        tyre_age=12,
        fuel_correction=fuel_correction
    )

    print(result)


main()