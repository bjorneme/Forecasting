import torch
import torch.nn as nn

class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(MLPModel, self).__init__()
        # Initialize parameters
        self.input_size = input_size * 24 # input_size is per time step. I assume 24 time steps
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size

        # Initialize Architecture MLP
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(self.input_size, self.hidden_layer_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(self.hidden_layer_size, self.hidden_layer_size)
        self.output = nn.Linear(self.hidden_layer_size, self.output_size)

    def forward(self, x):
        # Forward pass through the network.
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.output(x)
        return x
