import pandas as pd

import torch
from torch.utils.data import Dataset

from vol_predict.base.returns import Returns


class ReturnsDataset(Dataset):
    def __init__(
        self,
        returns: Returns,
        features: pd.DataFrame,
    ):
        self.dates = returns.log_returns.iloc[1:].to_numpy()
        self.returns = torch.Tensor(returns.log_returns.iloc[1:].to_numpy()).to(
            torch.float32
        )
        self.past_returns = torch.Tensor(
            returns.log_returns.shift(1).iloc[1:].to_numpy()
        ).to(torch.float32)
        self.features = torch.Tensor(features.iloc[1:].to_numpy()).to(torch.float32)

    def __len__(self):
        return len(self.returns)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.dates[idx],
            self.features[idx],
            self.past_returns[idx],
            self.returns[idx],
        )
