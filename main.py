from ForcastSystem import ForcastingSystem
from models.LSTMModel import LSTMModel
from models.CNNModel import CNNModel
from models.GRUModel import GRUNet

# The main function
if __name__ == "__main__":
    input_size = 4
    hidden_layer_size = 24
    output_size = 1
    num_layers = 2

    model = GRUNet(input_size, hidden_layer_size, output_size, num_layers)
    
    system = ForcastingSystem(
        filepath="consumption_and_temperatures.csv",
        area_number=1,
        model = model,
        num_epochs = 10,
        learning_rate = 0.001,
        model_filepath = "models/pre_trained_models/gru_model.pth"
    )
    system.run_system()
