import torch
import torch.nn as nn

# GRU Model
class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.gru = nn.GRU(input_dim, hidden_dim, layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.layers, x.size(0), self.hidden_dim).requires_grad_()
        out, _ = self.gru(x, h0.detach())
        out = self.fc(out[:, -1, :])  # Only take the output from the final timestep
        return out
