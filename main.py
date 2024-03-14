from ForcastSystem import ForcastingSystem
from models.LSTMModel import LSTMModel
from models.CNNModel import CNNModel
from models.GRUModel import GRUModel
from models.TransformerModel import TransformerModel

# The main function
if __name__ == "__main__":
    input_size = 4  # month, day, hour, temperature
    num_heads = 2
    num_encoder_layers = 2
    num_decoder_layers = 2
    dim_feedforward = 64
    max_seq_length = 24
    output_size = 1

    model = TransformerModel(input_size, num_heads, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length, output_size)
    
    system = ForcastingSystem(
        filepath="consumption_and_temperatures.csv",
        area_number=1,
        model = model,
        num_epochs = 1,
        learning_rate = 0.001,
        model_filepath = "models/pre_trained_models/transformer_model.pth"
    )
    system.run_system()
