"""
Low-Rank Adaptation (LoRA) for linear layers.

LoRA injects trainable low-rank matrices A and B into a frozen linear layer W:
  output = Wx + (B @ A)x * scaling

Only A and B are trained; W stays frozen. Import LoRALinear from here.
"""

import torch
from torch import nn


class LoRALinear(nn.Module):
    """
    Adds low-rank adaptation: output = Wx + (B @ A)x * scaling.

    Args:
        linear_layer: The original nn.Linear layer (will be frozen).
        rank: LoRA rank (typical: 4, 8, 16).
        alpha: Scaling factor for LoRA output (2*rank).
    """

    def __init__(self, linear_layer: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.linear = linear_layer
        in_features = linear_layer.weight.shape[1]
        out_features = linear_layer.weight.shape[0]
        self.rank = rank
        self.scaling = alpha / rank if rank > 0 else 0.0

        # LoRA matrices: ΔW = B @ A
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, rank))

        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

        # Freeze the original linear layer
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.linear(x)
        if self.rank > 0 and self.scaling != 0:
            # (batch, seq, in) @ (rank, in).T -> (batch, seq, rank)
            # (batch, seq, rank) @ (out, rank).T -> (batch, seq, out)
            lora_out = (x @ self.lora_A.T) @ self.lora_B.T
            base_out = base_out + lora_out * self.scaling
        return base_out
