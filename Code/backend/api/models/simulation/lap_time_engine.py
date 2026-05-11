import os
import pickle

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

base_pace_path = os.path.join(
    BASE_DIR,
    "saved_models",
    "track_base_pace.pkl"
)

deg_model_path = os.path.join(
    BASE_DIR,
    "saved_models",
    "all_tracks_degradation.pkl"
)

with open(base_pace_path, "rb") as f:
    track_base_pace = pickle.load(f)

with open(deg_model_path, "rb") as f:
    degradation_models = pickle.load(f)

print(track_base_pace.keys())
print(degradation_models.keys())