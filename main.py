from ForecastingSystem import ForecastingSystem
from models.LSTMModel import LSTMModel

# The main function
if __name__ == "__main__":

    system = ForecastingSystem(
        filepath="consumption_and_temperatures.csv",
        area_number=1,
        model=LSTMModel(input_size=4, hidden_layer_size=24, output_size=24),
        num_epochs = 2,
        learning_rate = 0.001,
        model_filepath = "models/pre_trained_models/lstm_model.pth"
    )
    system.run_system()
    system.visualize_learning_progress()
    system.visualize_dataset()