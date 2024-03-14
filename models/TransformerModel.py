import torch
import torch.nn as nn

# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, input_size, num_heads, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length, output_size):
        super(TransformerModel, self).__init__()
        self.input_size = input_size
        self.embed = nn.Linear(input_size, dim_feedforward)
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_seq_length, dim_feedforward))
        self.transformer = nn.Transformer(d_model=dim_feedforward, nhead=num_heads, 
                                          num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward, batch_first=True)
        self.out = nn.Linear(dim_feedforward, output_size)
        
    def forward(self, src):
        src = self.embed(src) + self.pos_encoder[:, :src.size(1), :]
        output = self.transformer(src, src)
        output = self.out(output[:, -1, :])
        return output