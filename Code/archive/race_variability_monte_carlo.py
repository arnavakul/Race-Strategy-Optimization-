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

from api.models.simulation.track_model import (
    get_track_parameters
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

    for sim in range(simulations):

        result = execute_race(

            track=track,

            starting_compound=starting_compound,

            total_laps=total_laps
        )

        race_times.append(
            result["total_time"]
        )

        safety_car_deployments.append(

            result[
                "safety_car_deployments"
            ]
        )

        vsc_deployments.append(

            result[
                "vsc_deployments"
            ]
        )

        if result["legal_race"]:

            legal_races += 1

    race_times.sort()

    average_time = statistics.mean(
        race_times
    )

    median_time = statistics.median(
        race_times
    )

    best_case = min(
        race_times
    )

    worst_case = max(
        race_times
    )

    std_dev = statistics.stdev(
        race_times
    )

    p5 = race_times[
        int(
            len(race_times) * 0.05
        )
    ]

    p95 = race_times[
        int(
            len(race_times) * 0.95
        )
    ]

    average_sc = statistics.mean(
        safety_car_deployments
    )

    average_vsc = statistics.mean(
        vsc_deployments
    )

    legality_rate = (
        legal_races
        / simulations
    )

    return {

        "simulations": simulations,

        "average_time": average_time,

        "median_time": median_time,

        "best_case": best_case,

        "worst_case": worst_case,

        "std_dev": std_dev,

        "p5": p5,

        "p95": p95,

        "average_sc": average_sc,

        "average_vsc": average_vsc,

        "legality_rate": legality_rate,

        "race_times": race_times
    }

def run_strategy_monte_carlo(
    strategy,
    track,
    simulations=100,
    seed=None
):

    if seed is not None:

        random.seed(seed)

    race_times = []

    simulation = simulate_strategy(

        race_context=race_context,

        strategy=strategy

    )

    # finish_positions = []
    
    for _ in range(simulations):

        race_context = {

            "track": track,

            "total_laps": sum(
                laps
                for _, laps in strategy
            ),

            "track_characteristics": get_track_parameters(track)

        }

        simulation = simulate_strategy(

            race_context=race_context,

            strategy=strategy

        )

        race_times.append(

            simulation["total_time"]
        )
        
        # finish_positions.append(
        #     simulation["final_position"]
        # )

    race_times.sort()

    average_time = statistics.mean(
        race_times
    )
    
    # average_finish = statistics.mean(
    #     finish_positions
    # )
    
    # podiums = len(

    #     [

    #         pos

    #         for pos

    #         in finish_positions

    #         if pos <= 3

    #     ]

    # )

    # podium_probability = (

    #         podiums

    #         / simulations

    #     ) * 100
        
    # wins = len(

    #     [

    #         pos

    #         for pos

    #         in finish_positions

    #         if pos == 1

    #     ]

    # )

    # win_probability = (

    #     wins

    #     / simulations

    # ) * 100

    median_time = statistics.median(
        race_times
    )

    best_case = min(
        race_times
    )

    worst_case = max(
        race_times
    )

    std_dev = statistics.stdev(
        race_times
    )

    p5 = race_times[
        int(
            len(race_times) * 0.05
        )
    ]

    p95 = race_times[
        int(
            len(race_times) * 0.95
        )
    ]

    return {

        "simulations": simulations,

        "average_time": average_time,

        "median_time": median_time,

        "best_case": best_case,

        "worst_case": worst_case,

        "std_dev": std_dev,

        "p5": p5,

        "p95": p95,

        "race_times": race_times,
        
        # "average_finish": average_finish,

        # "podium_probability":
        #     podium_probability,

        # "win_probability":
        #     win_probability,
            
    }
# TESTING

if __name__ == "__main__":

    print("\nSTARTING STRATEGY OPTIMIZER...\n")

    strategies = generate_strategies(57)

    print(
        f"Generated Strategies: "
        f"{len(strategies)}"
    )

    # Only test first few strategies while debugging
    strategies = strategies[:10]

    print(
        f"Testing First "
        f"{len(strategies)} "
        f"Strategies\n"
    )

    for i, strategy in enumerate(
        strategies,
        start=1
    ):

        print(
            f"\n[{i}/{len(strategies)}] "
            f"Testing Strategy:"
        )

        print(strategy)

        mc_result = run_strategy_monte_carlo(

            strategy=strategy,

            track="monza_2022",

            simulations=5
        )

        print(
            f"Average Time: "
            f"{mc_result['average_time']:.3f}"
        )

        print(
            f"Std Dev: "
            f"{mc_result['std_dev']:.3f}"
        )

        print("-" * 50)

    print(
        "\nDEBUG TEST COMPLETE\n"
    )