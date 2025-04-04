import pandas as pd

import torch
from torch.utils.data import Dataset

from vol_predict.base.returns import Returns
from vol_predict.features.base_preprocessor import BasePreprocessor


class ReturnsDataset(Dataset):
    def __init__(
        self,
        returns: Returns,
        vols: pd.DataFrame,
        features: pd.DataFrame,
        preprocessor: BasePreprocessor,
    ):
        self.preprocessor = preprocessor

        self.returns = returns.log_returns.iloc[1:]
        self.vols = vols.iloc[1:]
        self.past_returns = returns.log_returns.shift(1).iloc[1:]
        self.past_vols = vols.shift(1).iloc[1:]
        self.features = features.iloc[1:]

    def __len__(self):
        return len(self.returns)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.features.iloc[idx]
        past_returns = self.past_returns.iloc[idx]
        past_vols = self.past_vols.iloc[idx]
        true_returns = self.returns.iloc[idx]
        true_vols = self.vols.iloc[idx]

        features = self.preprocessor.transform(features)

        features = torch.Tensor(features.to_numpy()).to(torch.float32)
        past_returns = torch.Tensor([past_returns]).to(torch.float32)
        past_vols = torch.Tensor([past_vols]).to(torch.float32)

        true_returns = torch.Tensor([true_returns]).to(torch.float32)
        true_vols = torch.Tensor([true_vols]).to(torch.float32)

        return features, past_returns, past_vols, true_returns, true_vols
