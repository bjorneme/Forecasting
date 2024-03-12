from ForcastSystem import ForcastingSystem
from LSTMModel import LSTMModel

# The main function
if __name__ == "__main__":
    
    system = ForcastingSystem(
        filepath="consumption_and_temperatures.csv",
        area_number=1,
        model=LSTMModel()
    )
    system.run_system()