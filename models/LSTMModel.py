import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3,dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, self.num_layers, batch_first=True, dropout=dropout)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Linear layer for output
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state (short-term) and cell state (long-term)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        hidden = (h0, c0)
        
        # Forward pass LSTM layer
        lstm_out, _ = self.lstm(x, hidden)
        
        # Apply dropout
        dropout_out = self.dropout(lstm_out)
        
        # Prediction output
        predictions = self.linear(dropout_out[:, -1])
        
        return predictions
