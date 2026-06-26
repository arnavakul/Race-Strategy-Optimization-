from api.models.weekend.historical_degradation_lookup import (
    generate_weekend_degradation,
    load_degradation_models
)

def build_all_reports():

    models = load_degradation_models()

    tracks = set()

    for key in models.keys():

        track = "_".join(
            key.split("_")[:-1]
        )

        tracks.add(track)

    for track in sorted(tracks):

        print(
            f"\nBuilding {track}"
        )

        generate_weekend_degradation(
            track
        )

if __name__ == "__main__":

    build_all_reports()