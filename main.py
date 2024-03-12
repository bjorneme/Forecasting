from ForcastSystem import ForcastingSystem
from LSTMModel import LSTMModel

# The main function
if __name__ == "__main__":
    
    model = LSTMModel(
        input_dim=1, #TODO: Understand
        hidden_dim=10, #TODO: Understand
        output_dim=1, #TODO: Understand
        num_layers=2 # TODO: Understand
    )
    
    system = ForcastingSystem(
        filepath="consumption_and_temperatures.csv",
        area_number=1,
        model=model,
        num_epochs=25,
        learning_rate=10,
        model_filepath="lstm_model.pth"
    )
    system.run_system()