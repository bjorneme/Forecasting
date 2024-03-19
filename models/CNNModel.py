import torch
import torch.nn as nn

# CNN Model
class CNNModel(nn.Module):
    def __init__(self, sequence_length, output_size, dropout=0.5):
        super(CNNModel, self).__init__()

        # Convolution layer 1
        self.conv1 = nn.Conv1d(in_channels=sequence_length, out_channels=128, kernel_size=3, padding=1)
        
        # Activation function
        self.relu = nn.ReLU()

        # Convolution layer 2
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        
        # Flatten layer. Transform from 2D to 1D
        self.flatten = nn.Flatten()

        # Fully connected layer 1
        self.fc1 = nn.Linear(128 * 4, 512)

        # Dropout layer added after the first fully connected layer
        self.dropout = nn.Dropout(dropout)

        # Output layer
        self.fc2 = nn.Linear(512, output_size)

    def forward(self, x):

        # Forward pass through the first convolution layer
        x = self.relu(self.conv1(x))

        # Forward pass through the second convolution layer
        x = self.relu(self.conv2(x))
                      
        # Flatten the output for the fully connected layer.
        x = self.flatten(x)

        # Forward pass through the first fully connected layer
        x = self.dropout(self.relu(self.fc1(x)))

        # Output layer
        x = self.fc2(x)

        return x

