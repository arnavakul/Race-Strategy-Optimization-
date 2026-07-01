from typing import List, Dict, Any
import random

from api.models.simulation.lap_time_engine import (
    compute_lap_time
)

from api.models.simulation.fuel_state import (
    FuelState
)

from api.models.simulation.track_model import (
    get_track_parameters
)

from api.models.simulation.crossover_logic import (
    get_recommended_compound
)

from api.models.simulation.pitstop_model import (
    get_pitstop_time
)

from api.models.simulation.strategy_decision_engine import (
    should_pit
)

from api.models.simulation.race_state import (
    RaceState
)

# STINT SIMULATOR

def simulate_stint(
    track: str,
    stint_laps: int,
    race_laps: int,
    weather_timeline,
    race_state,
    starting_lap: int = 0,
    weekend_tyre_model=None
) -> Dict[str, Any]:

    # Store lap telemetry
    results: List[Dict[str, Any]] = []

    # Total stint time
    cumulative_time: float = 0.0

    # Initialize fuel model
    fuel = FuelState(
        starting_fuel=100,
        fuel_burn_per_lap=1.8,
    )

    # Track-specific parameters
    track_data = get_track_parameters(track)

    # Warmup penalties
    warmup_map = track_data["warmup_penalty"]

    # Simulate every lap
    for lap in range(stint_laps):

        # Global race lap
        global_lap = starting_lap + lap

        # Human-readable lap number
        current_lap = global_lap + 1

        # Current weather
        weather_state = weather_timeline[
            min(global_lap, len(weather_timeline) - 1)
        ]

        # Pit decision logic 
        pit_decision = should_pit(

            track=track,

            tyre_age=race_state.current_tyre_age,

            compound=race_state.current_compound,

            weather_state=weather_state,

            rival_gap=1.5,

            current_lap=current_lap,

            total_laps=race_laps,

            strategy_profile="BALANCED",

            safety_car_active=False,

            vsc_active=False
        )

        pit_for_weather = pit_decision["pit"]
        

        pit_loss:float = 0.0

        # Dynamic crossover pitstop
        if pit_for_weather:

            laps_remaining = race_laps - current_lap

            new_compound = (
                get_recommended_compound(
                    weather_state,
                    laps_remaining
                )
            )

            # Register new compound
            race_state.register_compound_usage(
                new_compound
            )
            
            race_state.current_compound = new_compound
            race_state.current_tyre_age = 0

            # Register pitstop
            race_state.register_pitstop()

            # Add pitloss
            pit_loss = get_pitstop_time(
                track = track
            )

            cumulative_time += pit_loss

            # Log event
            race_state.log_event(

                f"Lap {current_lap}: "

                f"{pit_decision['reason']} -> "

                f"{new_compound}"

            )

        

        # Default warmup
        warmup_penalty: float = 0.0

        # Warmup for first 2 laps
        if race_state.current_tyre_age <= 2:

            warmup_penalty = float(

                warmup_map[
                    race_state.current_compound
                ]
            )

        # Fuel correction
        fuel_correction: float = (
            fuel.getFuelCorrection()
        )

        # Compute lap physics
        lap_data: Dict[str, Any] = (

            compute_lap_time(

                track=track,

                compound=race_state.current_compound,

                tyre_age=race_state.current_tyre_age,

                current_lap=current_lap,

                total_laps=race_laps,

                fuel_correction=fuel_correction,

                weekend_tyre_model=weekend_tyre_model

            )

        )

        # Final lap time
        corrected_lap_time: float = (

            float(lap_data["lap_time"])

            + warmup_penalty
        )

        # Mixed weather slowdown
        if weather_state == "MIXED":

            corrected_lap_time += (
                random.uniform(0.5, 1.5)
            )

        # Wet weather slowdown
        elif weather_state == "WET":

            corrected_lap_time += (
                random.uniform(2, 5)
            )

        # Update stint time
        cumulative_time += corrected_lap_time

        # Store telemetry
        results.append({

            "lap": current_lap,

            "tyre_age": (
                race_state.current_tyre_age
            ),

            "lap_time": corrected_lap_time,

            "warmup_penalty": (
                warmup_penalty
            ),

            "fuel_load": float(
                fuel.current_fuel
            ),

            "fuel_correction": (
                fuel_correction
            ),

            "cumulative_time": (
                cumulative_time
            ),

            "weather_state": (
                weather_state
            ),

            "compound": (
                race_state.current_compound
            ),

            "pit_for_weather": (
                pit_for_weather
            ),

            "pit_loss": pit_loss
        })
        # Increment tyre age
        race_state.increment_tyre_age()
        
        # Burn fuel
        fuel.burnFuel()

    # Final output
    return {

        "total_time": cumulative_time,

        "laps": results
    }


# # TESTING


# if __name__ == "__main__":

#     weather_timeline = [

#         "DRY",
#         "DRY",
#         "DRY",
#         "DRY",
#         "DRY",

#         "MIXED",
#         "MIXED",
#         "MIXED",

#         "WET",
#         "WET",
#         "WET",

#         "DRY",
#         "DRY",
#         "DRY",
#         "DRY",

#         "DRY",
#         "DRY",
#         "DRY",
#         "DRY",
#         "DRY",

#         "DRY",
#         "DRY",
#         "DRY",
#         "DRY",
#         "DRY"
#     ]

#     race_state = RaceState()

#     race_state.register_compound_usage(
#         "MEDIUM"
#     )

#     result = simulate_stint(

    #     track="bahrain_2024",

    #     total_laps=25,

    #     weather_timeline=weather_timeline,

    #     race_state=race_state,

    #     weekend_tyre_model=None
    # )

#     print("\nSTINT SIMULATION\n")

#     previous_lap_time = None

#     for lap in result["laps"]:

#         if previous_lap_time is None:

#             delta = 0.0

#         else:

#             delta = (

#                 lap["lap_time"]

#                 - previous_lap_time
#             )

#         if lap["pit_for_weather"]:

#             print(
#                 f"\n=== PITSTOP ON LAP "
#                 f"{lap['lap']} ===\n"
#             )

#         print(

#             f"Lap {lap['lap']:>2} | "

#             f"Weather: {lap['weather_state']:<7} | "

#             f"Compound: {lap['compound']:<13} | "

#             f"Tyre Age: {lap['tyre_age']:>2} | "

#             f"Fuel: {lap['fuel_load']:.2f} kg | "

#             f"Lap Time: {lap['lap_time']:.3f} | "

#             f"Warmup: {lap['warmup_penalty']:.2f} | "

#             f"Pit: {lap['pit_for_weather']} | "

#             f"Pit Loss: {lap['pit_loss']:<5} | "

#             f"Delta: {delta:+.3f} | "

#             f"Cumulative: {lap['cumulative_time']:.3f}"
#         )

#         previous_lap_time = lap["lap_time"]

#     print(

#         f"\nTotal Stint Time: "

#         f"{result['total_time']:.3f}"
#     )