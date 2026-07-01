from api.models.simulation.stint_simulator import (
    simulate_stint
)

from api.models.simulation.pitstop_model import (
    get_pitstop_time
)

from api.models.simulation.race_state import (
    RaceState
)

from api.models.simulation.weather_model import (
    generate_weather_timeline
)


def simulate_strategy(
    race_context,
    strategy
):

    race_state = RaceState()

    track = race_context["track"]

    weekend_tyre_model = race_context.get(
        "weekend_tyre_model",
        {}
    )

    total_race_laps = sum(
        laps for _, laps in strategy
    )

    # Optional validation
    expected_laps = race_context["total_laps"]

    if (
        expected_laps is not None
        and total_race_laps != expected_laps
    ):

        raise ValueError(

            f"Strategy covers {total_race_laps} laps "

            f"but race requires {expected_laps}."

        )

    weather_timeline = generate_weather_timeline(
        total_race_laps
    )

    total_race_time = 0.0

    all_laps = []

    race_lap_cursor = 0

    for i, (compound, laps) in enumerate(strategy):

        race_state.current_compound = compound

        race_state.current_tyre_age = 0

        race_state.register_compound_usage(
            compound
        )

        stint_result = simulate_stint(

            track=track,

            stint_laps=laps,

            race_laps=total_race_laps,

            weather_timeline=weather_timeline,

            race_state=race_state,

            starting_lap=race_lap_cursor,

            weekend_tyre_model=weekend_tyre_model
        )

        race_lap_cursor += laps

        total_race_time += (
            stint_result["total_time"]
        )

        all_laps.extend(
            stint_result["laps"]
        )

        race_state.add_stint_record(

            compound=compound,

            laps=laps

        )

        is_final_stint = (
            i == len(strategy) - 1
        )

        if not is_final_stint:

            pit_loss = get_pitstop_time(
                track=track
            )

            total_race_time += pit_loss

            race_state.register_pitstop()

            race_state.log_event(

                f"Pitstop after stint {i + 1}"

            )

    race_state.validate_fia_legality()

    return {

        "strategy": strategy,

        "total_time": total_race_time,

        "pitstops": race_state.pitstop_count,

        "laps": all_laps,

        "stints": race_state.stint_history,

        "events": race_state.strategy_events,

        "legal_race": race_state.is_legal_race,

        "weather": weather_timeline

    }