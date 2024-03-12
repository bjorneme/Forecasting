from ForcastSystem import ForcastingSystem

# The main function
if __name__ == "__main__":
    
    system = ForcastingSystem(
        filepath="consumption_and_temperatures.csv",
        area_number=1
    )
    system.run_system()
