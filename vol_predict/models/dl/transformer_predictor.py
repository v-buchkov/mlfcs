from __future__ import annotations

import torch
import torch.nn as nn

from vol_predict.models.abstract_predictor import AbstractPredictor

def positional_encoding(x: torch.Tensor, device: str, n_levels: int) -> torch.Tensor:
    pos = torch.arange(0, n_levels, 1, dtype=torch.float32) / (n_levels - 1)
    pos = (pos + pos) - 1
    # pos = np.reshape(pos, (pos.shape[0]))
    pos_final = torch.zeros((x.shape[0], n_levels, 1), dtype=torch.float32)
    for i in range(pos_final.shape[0]):
        for j in range(pos_final.shape[1]):
            pos_final[i, j, 0] = pos[j]

    pos_final = torch.tensor(pos_final).to(device)
    x = torch.cat((x, pos_final), 2)

    return x


class TransformerPredictor(AbstractPredictor):
    def __init__(
        self,
        hidden_size: int,
        n_features: int,
        n_unique_features: int,
        n_layers: int,
        n_attention_heads: int,
        dim_feedforward: int,
        dropout: float,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.n_features = n_features
        self.n_unique_features = n_unique_features
        self.n_layers = n_layers

        self.sequence_length = self.n_features // self.n_unique_features

        torch.manual_seed(12)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_unique_features + 1, # +1, as have additional feature = position encoding
            nhead=n_attention_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(self.encoder_layer, n_layers)

        self.final_layer = nn.Sequential(
            nn.Linear((self.n_unique_features + 1) * self.sequence_length + 2, hidden_size), # +1 from positional encoding
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

        for name, param in self.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)  # Orthogonal initialization
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param)  # Xavier initialization

    def _forward(
        self,
        past_returns: torch.Tensor,
        past_vols: torch.Tensor,
        features: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        model_device = past_returns.device

        features = features[:, 12:]

        # Reshape features to have shape (batches, sequence, features)
        features = features.reshape(
            features.shape[0], self.sequence_length, self.n_unique_features
        )

        out = positional_encoding(features, model_device, self.sequence_length)

        # Pass the output from the previous steps through the transformer encoder
        out = self.transformer(out)

        out = torch.reshape(out, (out.shape[0], out.shape[1] * out.shape[2]))

        out = self.final_layer(torch.cat([out, past_returns, past_vols], dim=1))

        return nn.Softplus()(out)
