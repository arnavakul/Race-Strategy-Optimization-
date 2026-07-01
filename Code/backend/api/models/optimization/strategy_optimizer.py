from api.models.optimization.strategy_generator import (
    generate_strategies
)

from api.models.simulation.strategy_simulation import (
    simulate_strategy
)

from api.models.optimization.race_variability_monte_carlo import(
    run_strategy_monte_carlo
)

from api.models.simulation.race_constraints import(
    validate_weather_strategy
)

from api.models.simulation.weather_model import(
    generate_weather_state
)

def strategy_optimizer(
    race_context,
    risk_factor=0.5
):
    
    track = race_context["track"]
    
    weekend_tyre_model = race_context[
        "weekend_tyre_model"
    ]
    track_characteristics = race_context["track_characteristics"]

    total_laps = race_context["total_laps"]

    
    strategies = generate_strategies(total_laps)
    
    team_strength = race_context["team_strength"]

    print(
        f"Generated Strategies: "
        f"{len(strategies)}"
    )

    # DEBUG MODE
    strategies = strategies[:5]

    print(
        f"Testing "
        f"{len(strategies)} "
        f"strategies only"
)

    results = []

    for index, strategy in enumerate(
        strategies,
        start=1
    ):

        print(
            f"\nProcessing "
            f"{index}/{len(strategies)}"
        )
        
        weather_state = generate_weather_state()
        is_valid_weather_strategy = (
            validate_weather_strategy(
                strategy, weather_state
            )
        )
        
        if not is_valid_weather_strategy:
            continue
        
        # mc_result = run_strategy_monte_carlo(
        #     strategy=strategy,
        #     track = race_context["track"],
        #     simulations = 20
        # )
        
        mc_result = run_strategy_monte_carlo(

            strategy=strategy,

            race_context=race_context,

            simulations=20

        )
        
        average_time = mc_result["average_time"]
        
        std_dev = mc_result["std_dev"]
        
        risk_penalty = risk_factor * std_dev
        
        strategy_score = ( average_time + risk_penalty )

        simulation = simulate_strategy(
            race_context=race_context,
            strategy=strategy
        )

        average_lap = (
            simulation["total_time"]
            / total_laps
        )

        compounds_used = list(
            set(
                compound
                for compound, _ in strategy
            )
        )

        strategy_type = (
            f"{len(strategy)-1}-stop"
        )

        results.append({
            
            "average_time": average_time,
            
            "std_dev": std_dev,
            
            "strategy_score": strategy_score,

            "strategy": strategy,

            "strategy_type": strategy_type,

            "total_time": simulation["total_time"],

            "average_lap_time": average_lap,

            "pitstops": simulation["pitstops"],

            "stint_count": len(strategy),

            "compounds_used": compounds_used,

            "total_laps": total_laps
        })

    sorted_results = sorted(
        results,
        key=lambda x: x["strategy_score"]
    )

    unique_strategies = {}

    for result in sorted_results:

        strategy_key = tuple(result["strategy"])

        if strategy_key not in unique_strategies:

            unique_strategies[strategy_key] = result

    return list(
        unique_strategies.values()
    )[:20]


# #TESTING BLOCK 
# if __name__ == "__main__":

#     print("\nSTARTING STRATEGY OPTIMIZER\n")

#     strategies = generate_strategies(57)

#     print(
#         f"Generated Strategies: "
#         f"{len(strategies)}"
#     )

#     # DEBUG MODE
#     strategies = strategies[:20]

#     print(
#         f"Testing First "
#         f"{len(strategies)} "
#         f"Strategies Only\n"
#     )

#     for i, strategy in enumerate(
#         strategies,
#         start=1
#     ):

#         print(
#             f"\n[{i}/{len(strategies)}]"
#         )

#         print(
#             f"Strategy: {strategy}"
#         )

#         try:

#             mc_result = run_strategy_monte_carlo(

#                 strategy=strategy,

#                 track="monza_2022",

#                 simulations=3
#             )

#             print(
#                 f"Average Time: "
#                 f"{mc_result['average_time']:.3f}"
#             )

#             print(
#                 f"Std Dev: "
#                 f"{mc_result['std_dev']:.3f}"
#             )

#             print(
#                 f"Best Case: "
#                 f"{mc_result['best_case']:.3f}"
#             )

#             print(
#                 f"Worst Case: "
#                 f"{mc_result['worst_case']:.3f}"
#             )

#         except Exception as e:

#             print(
#                 f"FAILED: {e}"
#             )

#         print("-" * 50)

#     print(
#         "\nDEBUG RUN COMPLETE\n"
#     )