import pandas as pd

import torch
from torch.utils.data import Dataset

from vol_predict.base.returns import Returns
from vol_predict.features.base_preprocessor import BasePreprocessor


class ReturnsDataset(Dataset):
    def __init__(
        self,
        returns: Returns,
        features: pd.DataFrame,
        preprocessor: BasePreprocessor,
    ):
        self.preprocessor = preprocessor

        self.returns = returns.log_returns.iloc[1:]
        self.past_returns = returns.log_returns.shift(1).iloc[1:]
        self.features = features.iloc[1:]

    def __len__(self):
        return len(self.returns)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.features.iloc[idx]
        returns = self.returns.iloc[idx]
        past_returns = self.past_returns.iloc[idx]

        features = self.preprocessor.transform(features)

        features = torch.Tensor(features.to_numpy()).to(torch.float32)
        returns = torch.Tensor(returns.to_numpy()).to(torch.float32)
        past_returns = torch.Tensor(past_returns.to_numpy()).to(torch.float32)

        return features, returns, past_returns
