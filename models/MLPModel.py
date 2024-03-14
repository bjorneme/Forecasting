import torch
import torch.nn as nn

# Define the MLP model
class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_dim * 24, hidden_dim)  # Adjust input dimension for flat input
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input for MLP
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x