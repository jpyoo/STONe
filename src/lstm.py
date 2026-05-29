import torch
import torch.nn as nn
    
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size, bias=True)
    
    def forward(self, x):
        B = x.size(0)
        h0 = torch.zeros(self.num_layers, B, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, B, self.hidden_size, device=x.device)

        out, _ = self.lstm(x, (h0, c0))  # out: (B, T, H)
        last = out[:, -1, :]  # last time step: (B, H)
        last = self.layer_norm(last)
        out = self.fc(last)   # (B, output_size)
        return out
