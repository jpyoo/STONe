import torch
import torch.nn as nn
from fcn import FCN

class DeepONet(nn.Module):
    """
    Classical DeepONet (FCN-FCN) implementation using operator-style formulation.
    branch_arch: list of layer sizes for branch network, e.g., [input_dim, 64, 128]
    trunk_arch: list of layer sizes for trunk network, final layer size must be branch_output_size * num_outputs
    num_outputs: number of output channels per trunk query (e.g., temperature, pressure, etc.)
    """

    def __init__(self, branch_arch, trunk_arch, num_outputs=1, activation_fn=nn.ReLU):
        super(DeepONet, self).__init__()

        self.num_outputs = num_outputs
        self.branch_output_size = branch_arch[-1]

        # Build branch and trunk networks
        self.branch_net = FCN(branch_arch, activation_fn)

        # Automatically append output layer size to trunk_arch
        trunk_output_size = self.branch_output_size * self.num_outputs
        full_trunk_arch = trunk_arch + [trunk_output_size]

        self.trunk_net = FCN(full_trunk_arch, activation_fn)

    def forward(self, branch_input, trunk_input):
        """
        branch_input: Tensor of shape (B, D_branch)
        trunk_input: Tensor of shape (B, P, D_trunk)
        """
        B, P, _ = trunk_input.shape
        d = self.branch_output_size
        C = self.num_outputs

        # Branch: (B, d)
        branch_output = self.branch_net(branch_input)

        # Trunk: process each sample separately
        trunk_output = []
        for i in range(B):
            trunk_output.append(self.trunk_net(trunk_input[i]))  # (P, d * C)
        trunk_output = torch.stack(trunk_output)  # (B, P, d * C)

        # Reshape to (B, P, d, C)
        trunk_output = trunk_output.view(B, P, d, C)

        # Combine via einsum → (B, P, C)
        output = torch.einsum('bd,bpdc->bpc', branch_output, trunk_output)

        return output  # (B, P, C)
