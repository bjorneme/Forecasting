from ForecastingSystem import ForecastingSystem
from models.LSTMModel import LSTMModel

# The main function
if __name__ == "__main__":
    input_size = 4  # month, day, hour, temperature

    model = LSTMModel(input_size, 24, 1, 2)
    
    system = ForecastingSystem(
        filepath="consumption_and_temperatures.csv",
        area_number=1,
        model = model,
        num_epochs = 4,
        learning_rate = 0.001,
        model_filepath = "models/pre_trained_models/lstm_model.pth"
    )
    system.run_system()

    # Example usage
    model_paths = {
        "LSTM": "models/pre_trained_models/lstm_model.pth",
    }

    models = {
            "LSTM": LSTMModel(4, 24, 1, 2),
    }

    system.evaluate_and_plot_models(model_paths, models)
