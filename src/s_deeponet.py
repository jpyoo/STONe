import torch
import torch.nn as nn
import torch.nn.functional as F
from fcn import FCN, B_FCN
from lstm import LSTM
from gru import GRU
from rnn import RNN
from transformer import Transformer  # Assuming you have a Transformer class defined

class SequentialDeepONet(nn.Module):
    def __init__(self, branch_type, branch_input_size, branch_hidden_size, branch_num_layers, branch_output_size, 
                 trunk_architecture, num_outputs, activation_fn=nn.ReLU, pred_window = 1):
        super(SequentialDeepONet, self).__init__()

        self.num_outputs = num_outputs
        self.branch_output_size = branch_output_size * pred_window
        self.pred_window = pred_window

        # Branch network
        if branch_type == 'rnn':
            self.branch_net = RNN(branch_input_size, branch_hidden_size, branch_num_layers, self.branch_output_size)
        elif branch_type == 'lstm':
            self.branch_net = LSTM(branch_input_size, branch_hidden_size, branch_num_layers, self.branch_output_size)
        elif branch_type == 'gru':
            self.branch_net = GRU(branch_input_size, branch_hidden_size, branch_num_layers, self.branch_output_size)
        elif branch_type == 'fcn':  # New FCN-based branch option
            self.branch_net = B_FCN([branch_input_size] + [branch_hidden_size] * (branch_num_layers - 1) + [self.branch_output_size], activation_fn)
        elif branch_type == 'transformer':
            num_heads = 8  # Default number of heads
            self.branch_net = Transformer(branch_input_size, branch_hidden_size, num_heads, branch_num_layers, self.branch_output_size)
        else:
            raise ValueError(f"Unsupported branch type: {branch_type}")

        # Trunk network (output d * num_outputs * pred_window)
        trunk_output_size = branch_output_size * num_outputs
        self.trunk_net = FCN(trunk_architecture + [trunk_output_size], activation_fn)
        
        self.bias = nn.Parameter(torch.zeros(1, 1, 1, self.num_outputs))  # Shape: (1, 1, 1, C)

    def forward(self, branch_input, trunk_input):
        B, P, trunk_dim = trunk_input.shape
        d = self.branch_output_size // self.pred_window
        w = self.pred_window

        # Branch output: (B, d * w) Batch, hidden_dim, pred_window
        branch_output = self.branch_net(branch_input)
        # Reshape branch output to (B, w, d)
        branch_output = branch_output.view(B, w, d)  # (B, w, d)
        
        # Trunk output: (B, P, d * C)
        trunk_output = self.trunk_net(trunk_input)  # (B, P, d * C)
        # Reshape trunk output to (B, P, d, C)
        trunk_output = trunk_output.view(B, -1, d, self.num_outputs)  # (B, P, d, C)
        
        # DeepONet product: (B, P, C, w)
        output = torch.einsum('bwd,bpdc->bwpc', branch_output, trunk_output)

        return output + self.bias # shape: (B, pred_window, P, num_outputs)