import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, sequence_length, output_size):
        super(CNNModel, self).__init__()
        # Adjust the in_channels to match the actual number of channels in your input data
        self.conv1 = nn.Conv1d(in_channels=sequence_length, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        # Adjust the input feature size to the flattened output of the last conv layer
        self.fc1 = nn.Linear(128 * 4, 512)  # Assuming 4 is the correct sequence length after conv layers
        self.fc2 = nn.Linear(512, output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

