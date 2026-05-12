import pandas as pd
import os
import pickle
import glob

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

processed_data_path = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

output_path = os.path.join(
    BASE_DIR,
    "models",
    "saved_models",
    "track_base_pace.pkl"
)

parquet_files = glob.glob(
    os.path.join(processed_data_path,"*.parquet")
)

# print(parquet_files)
# print(processed_data_path)

track_base_pace = {}

for file in parquet_files:
    df = pd.read_parquet(file)
    track_name = os.path.basename(file).replace(".parquet","")
    print(df.columns)
    
    df = df[df["LapTimeSeconds"].notna()]
    
    sorted_laps = df.sort_values("LapTimeSeconds")
    
    top_n = max(1, int(len(sorted_laps)*0.10))
    
    best_laps = sorted_laps.head(top_n)
    
    base_pace = best_laps["LapTimeSeconds"].median()
    
    track_base_pace[track_name] = round(base_pace,3)
    
    print(f"{track_name}: {base_pace}")

with open(output_path, "wb")as f:
    pickle.dump(track_base_pace, f)
