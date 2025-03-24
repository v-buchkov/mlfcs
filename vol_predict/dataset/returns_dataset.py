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

        self.dates = returns.log_returns.iloc[1:].to_numpy()
        self.returns = returns.log_returns.iloc[1:].to_numpy()
        self.past_returns = returns.log_returns.shift(1).iloc[1:].to_numpy()
        self.features = features.iloc[1:].to_numpy()

    def __len__(self):
        return len(self.returns)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dates = self.dates[idx]
        features = self.features[idx]
        returns = self.returns[idx]
        past_returns = self.past_returns[idx]

        features = self.preprocessor.transform(features)

        dates = torch.Tensor(dates).to(torch.float32)
        features = torch.Tensor(features).to(torch.float32)
        returns = torch.Tensor(returns).to(torch.float32)
        past_returns = torch.Tensor(past_returns).to(torch.float32)

        return dates, features, returns, past_returns
