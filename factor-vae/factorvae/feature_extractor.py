import torch
from torch import nn


class FeatureExtractor(nn.Module):
    """Feature extractor for FactorVAE

    Args:
        num_chars (int): number of characteristics (input features)
        num_feats (int): dimension of features (output features)
    """

    def __init__(self, num_chars: int, num_feats: int):
        super().__init__()
        self.num_feats = num_feats
        self.linear_layer = nn.Sequential(
            nn.Linear(num_chars, num_chars),
            nn.LeakyReLU()
        )
        self.gru_cell = GRUCell3d(num_chars, num_feats)

    def forward(self, chars: torch.Tensor) -> torch.Tensor:
        # chars shape: [batch_size, len_hist, num_stocks, num_features]
        chars = chars.permute(1, 0, 2, 3)
        feats = torch.zeros(
            chars.shape[1],  # batch_size
            chars.shape[2],  # num_stocks
            self.num_feats,  # num_feats
            device=chars.device,
            dtype=chars.dtype
        )
        for char in chars:
            feats = self.gru_cell(char, feats)
        return feats


class GRUCell3d(nn.Module):
    """GRU cell applied over batch and stock dimensions."""

    def __init__(self, num_chars: int, num_feats: int):
        super().__init__()
        self.gru_cell = nn.GRUCell(num_chars, num_feats)

    def forward(self, data: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        # data shape: [batch_size, num_stocks, num_chars]
        # hidden shape: [batch_size, num_stocks, num_feats]
        batch_size, num_stocks, num_chars = data.shape
        _, _, num_feats = hidden.shape

        # Flatten batch and stock dimensions
        data_flat = data.reshape(-1, num_chars)        # [batch_size * num_stocks, num_chars]
        hidden_flat = hidden.reshape(-1, num_feats)    # [batch_size * num_stocks, num_feats]

        # Apply GRU cell
        output_flat = self.gru_cell(data_flat, hidden_flat)  # [batch_size * num_stocks, num_feats]

        # Reshape back to [batch_size, num_stocks, num_feats]
        output = output_flat.view(batch_size, num_stocks, num_feats)
        return output
