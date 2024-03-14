from ForcastSystem import ForcastingSystem
from models.LSTMModel import LSTMModel
from models.CNNModel import CNNModel
from models.GRUModel import GRUModel

# The main function
if __name__ == "__main__":
    input_channels = 4  # As before, for month, day, hour, temperature
    num_features = 24  # Number of temporal features after flattening
    output_size = 1

    model = CNNModel(input_channels, num_features, output_size)
    
    system = ForcastingSystem(
        filepath="consumption_and_temperatures.csv",
        area_number=1,
        model = model,
        num_epochs = 10,
        learning_rate = 0.001,
        model_filepath = "models/pre_trained_models/cnn_model.pth"
    )
    system.run_system()
