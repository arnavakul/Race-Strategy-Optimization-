from api.models.simulation.race_constraints import(
    validate_strategy
)

# COMPOUNDS = ["SOFT", "MEDIUM", "HARD","INTERMEDIATE","WET"]

DRY_COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]

MIN_STINT = {
    "SOFT": 10,
    "MEDIUM": 15,
    "HARD": 20,
    # "INTERMEDIATE":5,
    # "WET":3
}

MAX_STINT = {
    "SOFT": 20,
    "MEDIUM": 30,
    "HARD": 40,
    # "INTERMEDIATE":25,
    # "WET":20
}


def generate_strategies(total_laps):

    strategies = []
    
    strategy_set = set()

    # 1-stop

    for compound1 in DRY_COMPOUNDS:

        for compound2 in DRY_COMPOUNDS:

            if compound1 == compound2:
                continue

            for stint1_laps in range(
                MIN_STINT[compound1],
                MAX_STINT[compound1] + 1,3
            ):

                stint2_laps = (
                    total_laps
                    - stint1_laps
                )

                if not (
                    MIN_STINT[compound2]
                    <= stint2_laps
                    <= MAX_STINT[compound2]
                ):
                    continue

                strategy = [
                    (compound1, stint1_laps),
                    (compound2, stint2_laps)
                ]
                
                if validate_strategy(
                    strategy,
                    total_laps
                ):
                    
                    strategy_key = tuple(strategy)

                    if strategy_key not in strategy_set:

                        strategy_set.add(strategy_key)

                        strategies.append(strategy)

    # 2-stop
    
    STINT_STEP = 3

    for compound1 in DRY_COMPOUNDS:

        for compound2 in DRY_COMPOUNDS:

            for compound3 in DRY_COMPOUNDS:

                used_compounds = {
                    compound1,
                    compound2,
                    compound3
                }

                if len(used_compounds) < 2:
                    continue

                for stint1 in range(
                    MIN_STINT[compound1],
                    MAX_STINT[compound1] + 1,
                    STINT_STEP
                ):

                    for stint2 in range(
                        MIN_STINT[compound2],
                        MAX_STINT[compound2] + 1,
                        STINT_STEP
                    ):

                        stint3 = (
                            total_laps
                            - stint1
                            - stint2
                        )

                        if not (
                            MIN_STINT[compound3]
                            <= stint3
                            <= MAX_STINT[compound3]
                        ):
                            continue

                        strategy = [
                            (compound1, stint1),
                            (compound2, stint2),
                            (compound3, stint3)
                        ]
                        
                        if validate_strategy(
    strategy,
    total_laps
):
                            strategy_key = tuple(strategy)

                            if strategy_key not in strategy_set:

                                strategy_set.add(strategy_key)

                                strategies.append(strategy)

    strategies.sort(
        key=len
    )

    return strategies


if __name__ == "__main__":

    strategies = generate_strategies(57)

    print(f"\nGenerated {len(strategies)} strategies\n")

    for s in strategies[:20]:
        print(s)