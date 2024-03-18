import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size, dropout_rate=0.3):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        
        # First LSTM layer
        self.lstm1 = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        
        # Dropout after the first LSTM layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Second LSTM layer
        self.lstm2 = nn.LSTM(hidden_layer_size, hidden_layer_size, batch_first=True)
        
        # Linear layer to produce the final output
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        # Pass data through the first LSTM layer
        lstm1_out, _ = self.lstm1(x)
        
        # Apply dropout
        dropout_out = self.dropout(lstm1_out)
        
        # Pass data through the second LSTM layer
        lstm2_out, _ = self.lstm2(dropout_out)
        
        # Use the output of the last time step from the second LSTM layer
        predictions = self.linear(lstm2_out[:, -1])
        
        return predictions
