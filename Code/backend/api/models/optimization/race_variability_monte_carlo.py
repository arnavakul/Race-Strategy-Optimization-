import random
import statistics

from api.models.simulation.race_executor import (
    execute_race
)

from api.models.simulation.strategy_simulation import (
    simulate_strategy
)

from api.models.optimization.strategy_generator import (
    generate_strategies
)


def run_monte_carlo(
    track,
    starting_compound,
    total_laps,
    simulations=100,
    seed=None
):

    if seed is not None:
        random.seed(seed)

    race_times = []

    safety_car_deployments = []

    vsc_deployments = []

    legal_races = 0

    for _ in range(simulations):

        result = execute_race(

            track=track,

            starting_compound=starting_compound,

            total_laps=total_laps

        )

        race_times.append(
            result["total_time"]
        )

        safety_car_deployments.append(

            result["safety_car_deployments"]

        )

        vsc_deployments.append(

            result["vsc_deployments"]

        )

        if result["legal_race"]:

            legal_races += 1

    race_times.sort()

    return {

        "simulations": simulations,

        "average_time": statistics.mean(race_times),

        "median_time": statistics.median(race_times),

        "best_case": min(race_times),

        "worst_case": max(race_times),

        "std_dev": (
            statistics.stdev(race_times)
            if len(race_times) > 1
            else 0
        ),

        "p5": race_times[
            int(len(race_times) * 0.05)
        ],

        "p95": race_times[
            int(len(race_times) * 0.95)
        ],

        "average_sc": statistics.mean(
            safety_car_deployments
        ),

        "average_vsc": statistics.mean(
            vsc_deployments
        ),

        "legality_rate": (
            legal_races
            / simulations
        ),

        "race_times": race_times

    }


def run_strategy_monte_carlo(
    strategy,
    race_context,
    simulations=100,
    seed=None
):

    if seed is not None:

        random.seed(seed)

    race_times = []

    for _ in range(simulations):

        simulation = simulate_strategy(

            race_context=race_context,

            strategy=strategy

        )

        race_times.append(

            simulation["total_time"]

        )

    race_times.sort()

    return {

        "simulations": simulations,

        "average_time": statistics.mean(
            race_times
        ),

        "median_time": statistics.median(
            race_times
        ),

        "best_case": min(
            race_times
        ),

        "worst_case": max(
            race_times
        ),

        "std_dev": (
            statistics.stdev(race_times)
            if len(race_times) > 1
            else 0
        ),

        "p5": race_times[
            int(len(race_times) * 0.05)
        ],

        "p95": race_times[
            int(len(race_times) * 0.95)
        ],

        "race_times": race_times

    }


# ---------------- TESTING ---------------- #

if __name__ == "__main__":

    print("\nSTARTING STRATEGY MONTE CARLO\n")

    strategies = generate_strategies(57)

    print(
        f"Generated {len(strategies)} strategies"
    )

    race_context = {

        "track": "monza_2022",

        "total_laps": 57,

        "track_characteristics": {}

    }

    for strategy in strategies[:5]:

        print("\n", strategy)

        result = run_strategy_monte_carlo(

            strategy=strategy,

            race_context=race_context,

            simulations=5

        )

        print(result["average_time"])

    print("\nDONE\n")