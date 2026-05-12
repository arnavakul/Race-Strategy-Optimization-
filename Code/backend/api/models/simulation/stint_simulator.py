from lap_time_engine import compute_lap_time

results = []

def simulate_stint(
    track,
    compound,
    total_laps
):
    results = []
    cumulative_time = 0
    
    for lap in range(total_laps):
        tyre_age =lap
        current_lap = lap+1
        
        lap_data = compute_lap_time(
            track=track,
            compound=compound,
            tyre_age=tyre_age,
            current_lap=current_lap
        )
        
        cumulative_time += lap_data["lap_time"]
        
        results.append({
            "lap": current_lap,
            "tyre_age": tyre_age,
            "lap_time": lap_data["lap_time"],
            "cumulative_time": cumulative_time
        })
        
    return results

result = simulate_stint( #this is supposedly dynamic meaning that the person choosing will choose these. 
    track="bahrain_2022",
    compound="MEDIUM",
    total_laps=15
)

print("Sting Simulation:")
previous_lap_time = None

for lap in result:
    if previous_lap_time is None:
        delta = 0
    else:
        delta = lap["lap_time"] - previous_lap_time
    print(
        f"Lap {lap['lap']:>2} | "
        f"Tyre Age: {lap['tyre_age']:>2} | "
        f"Lap Time: {lap['lap_time']:.3f} | "
        f"Delta Lap Time: {delta:+.3f}| "
        f"Cumulative: {lap['cumulative_time']:.3f}"
    )
    previous_lap_time = lap["lap_time"]