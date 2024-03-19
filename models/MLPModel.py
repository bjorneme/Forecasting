import torch
import torch.nn as nn

import torch
import torch.nn as nn

class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, forcast_range, dropout_rate=0.5):
        super(MLPModel, self).__init__()
        # Flatten the input
        layers = [nn.Flatten()]
        
        # Setup the first layer input size
        input_size = input_size * forcast_range  # Assuming input_size is per time step and there are 24 time steps
        
        # Dynamically add hidden layers
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))

            # Apply ReLU
            layers.append(nn.ReLU())

            # Apply Dropout
            layers.append(nn.Dropout(dropout_rate))
            input_size = hidden_size  # Update last layer size for next layer
        
        # Add the output layer (no dropout before the output)
        layers.append(nn.Linear(input_size, output_size))
        
        # Combine all layers into a ModuleList for sequential processing
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # Pass the input through all layers
        return self.layers(x)
