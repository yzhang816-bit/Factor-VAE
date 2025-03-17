from typing import Tuple

import torch
from torch import nn


class FactorDecoder(nn.Module):
    """Factor decoder for FactorVAE

    Args:
        num_facts (int): number of facts $K$
        num_feats (int): number of features $H$
    """

    def __init__(self, num_facts: int, num_feats: int):
        super().__init__()
        self.alpha_layer = AlphaLayer(num_feats)
        self.beta_layer = nn.Linear(num_feats, num_facts)

    def forward(self, facts: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:

        return torch.einsum("bsk,bk->bs", self.beta_layer(feats), facts) + torch.normal(
            *self.alpha_layer(feats)
        )


class AlphaLayer(nn.Module):
    def __init__(self, num_feats: int):
        super().__init__()
        self.linear_layer = nn.Sequential(
            nn.Linear(num_feats, num_feats), nn.LeakyReLU()
        )
        self.mean_layer = nn.Linear(num_feats, 1)
        self.std_layer = nn.Sequential(nn.Linear(num_feats, 1), nn.Softplus())

    def forward(self, feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.linear_layer(feats)
        mean, std = self.mean_layer(hidden), self.std_layer(hidden)
        std = std.flatten(-2).clip(min=0).detach().cpu().apply_(lambda x: x if x > 0 else 1e-6)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Example of moving tensors to device
        std = std.to(device)
#       std = std.to(0)
        return mean.flatten(-2), std
