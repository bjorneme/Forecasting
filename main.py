from ForecastingSystem import ForecastingSystem
from models.LSTMModel import LSTMModel
from models.CNNModel import CNNModel
from models.GRUModel import GRUModel
from models.MLPModel import MLPModel
from models.TransformerModel import TransformerModel

# The main function
if __name__ == "__main__":
    input_size = 4  # month, day, hour, temperature

    model = LSTMModel(4, 24, 1, 2)
    
    system = ForecastingSystem(
        filepath="consumption_and_temperatures.csv",
        area_number=1,
        model = model,
        num_epochs = 0,
        learning_rate = 0.001,
        model_filepath = "models/pre_trained_models/lstm_model.pth"
    )
    system.run_system()

    # Example usage
    model_paths = {
        "LSTM": "models/pre_trained_models/lstm_model.pth",
        "MLP": "models/pre_trained_models/mlp_model.pth",
        # Add other models accordingly
    }

    models = {
            "LSTM": LSTMModel(4, 24, 1, 2),
            "MLP": MLPModel(4, 32, 1),
    }

    system.evaluate_and_plot_models(model_paths, models)
