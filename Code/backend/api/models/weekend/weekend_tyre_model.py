from copy import deepcopy


def build_weekend_tyre_model(

    historical_report,

    live_report,

    historical_weight=0.30,

    live_weight=0.70

):

    weekend_model = {}

    compounds = [

        "SOFT",

        "MEDIUM",

        "HARD"

    ]

    for compound in compounds:

        history = historical_report["degradation"][compound]

        historical_deg = (

            history["lap_15"]

            -

            history["lap_5"]

        ) / 10

        live_deg = live_report[compound]["degradation"]

        if live_deg is None:

            live_deg = historical_deg

        weekend_deg = (

            historical_weight * historical_deg

            +

            live_weight * live_deg

        )

        weekend_model[compound] = {

            "historical":

                round(historical_deg,4),

            "live":

                round(live_deg,4),

            "weekend":

                round(weekend_deg,4)

        }

    return weekend_model

if __name__ == "__main__":

    from api.models.weekend.historical_degradation_lookup import (
        generate_weekend_degradation
    )

    from api.models.weekend.weekend_live_degradation_builder import (
        load_weekend_data,
        filter_long_run_laps,
        build_live_degradation_report
    )


    # -------------------------------
    # Load Barcelona FP sessions
    # -------------------------------

    weekend_df = load_weekend_data(

        r"data\processed\FP1\clean_laps\barcelona_2026_FP1_clean.parquet",

        r"data\processed\FP2\clean_laps\barcelona_2026_FP2_clean.parquet",

        r"data\processed\FP3\clean_laps\barcelona_2026_FP3_clean.parquet"

    )

    weekend_df = filter_long_run_laps(
        weekend_df
    )


    # -------------------------------
    # Historical degradation
    # -------------------------------

    historical = generate_weekend_degradation(
        "barcelona"
    )


    # -------------------------------
    # Live degradation
    # -------------------------------

    live = build_live_degradation_report(
        weekend_df
    )


    # -------------------------------
    # Weekend tyre model
    # -------------------------------

    weekend_model = build_weekend_tyre_model(

        historical,

        live

    )


    print("\n========== HISTORICAL ==========\n")

    print(historical)


    print("\n========== LIVE ==========\n")

    print(live)


    print("\n========== WEEKEND TYRE MODEL ==========\n")

    for compound, data in weekend_model.items():

        print(

            f"{compound}\n"

            f"Historical : {data['historical']:.4f}\n"

            f"Live       : {data['live']:.4f}\n"

            f"Weekend    : {data['weekend']:.4f}\n"

        )