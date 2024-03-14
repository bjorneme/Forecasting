import torch
import torch.nn as nn

# CNN Model
class CNNModel(nn.Module):
    def __init__(self, input_channels, num_features, output_size):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(32 * num_features, output_size)
        
    def forward(self, x):
        # Transpose the input so that the channels come first
        x = x.transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.linear(x)
        return x