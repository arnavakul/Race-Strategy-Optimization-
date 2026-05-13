class FuelState: 
    
    def __init__(self,starting_fuel, fuel_burn_per_lap, fuel_effect_per_kg):
        self.current_fuel = starting_fuel
        self.fuel_burn_per_lap = fuel_burn_per_lap
        self.fuel_effect_per_kg = fuel_effect_per_kg
    
    def getFuelCorrection(self):
        return(
            self.current_fuel * self.fuel_effect_per_kg
        )
    
    def burnFuel(self):
        self.current_fuel -= self.fuel_burn_per_lap
        
        if self.current_fuel < 0:
            self.current_fuel = 0 


fuel = FuelState(
    starting_fuel=100,
    fuel_burn_per_lap=1.8,
    fuel_effect_per_kg=0.035
)

for lap in range(5):

    print(
        f"Lap {lap+1} | "
        f"Fuel: {fuel.current_fuel:.2f} kg | "
        f"Correction: {fuel.getFuelCorrection():.3f}"
    )

    fuel.burnFuel()