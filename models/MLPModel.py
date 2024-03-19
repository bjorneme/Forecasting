import torch
import torch.nn as nn

# MLP Model
class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, forcast_range, dropout_rate=0.5):
        super(MLPModel, self).__init__()
        # Flatten the input
        layers = [nn.Flatten()]
        
        # Input into the model
        input_size = input_size * forcast_range
        
        # Add hidden layers
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))

            # Apply ReLU
            layers.append(nn.ReLU())

            # Apply Dropout
            layers.append(nn.Dropout(dropout_rate))

            # Update input for next layer
            input_size = hidden_size
        
        # Add the output layer
        layers.append(nn.Linear(input_size, output_size))
        
        # Combine all layers
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # Forward Pass
        return self.layers(x)
