from ForecastingSystem import ForecastingSystem
from models.CNNModel import CNNModel
from models.LSTMModel import LSTMModel
from models.MLPModel import MLPModel

# The main function
if __name__ == "__main__":

    system = ForecastingSystem(
        filepath="consumption_and_temperatures.csv",
        area_number=1,
        forcast_range=24,
        num_lags=24,
        # model=LSTMModel(input_size=4, hidden_layer_size=100, output_size=24),
        # model=MLPModel(input_size=4, hidden_layers=[1024, 1024, 1024, 1024, 1024], output_size=24, forcast_range=24),
        model=CNNModel(sequence_length=24, output_size=24),
        num_epochs = 10,
        learning_rate = 0.001,
        # model_filepath = "models/pre_trained_models/lstm_model.pth"
        model_filepath = "models/pre_trained_models/cnn_model.pth"
        # model_filepath = "models/pre_trained_models/cnn_model.pth"
    )
    system.run_system()
    # system.visualize_learning_progress()
    # system.visualize_dataset()

    # ========================================
    # Used for comparing the models
    model_paths = {
        "LSTM": "models/pre_trained_models/lstm_model.pth",
        "MLP": "models/pre_trained_models/mlp_model.pth",
        "CNN": "models/pre_trained_models/cnn_model.pth",
    }

    models = {
            "LSTM": LSTMModel(input_size=4, hidden_layer_size=100, output_size=24),
            "MLP": MLPModel(input_size=4, hidden_layers=[1024, 1024, 1024, 1024, 1024], output_size=24, forcast_range=24),
            "CNN": CNNModel(sequence_length=24, output_size=24),
    }

    system.evaluate_and_plot_models(model_paths, models)
