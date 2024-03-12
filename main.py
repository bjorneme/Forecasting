from ForcastSystem import ForcastingSystem

# The main function
if __name__ == "__main__":
    
    system = ForcastingSystem(filepath="consumption_and_temperatures.csv")
    system.run_system()