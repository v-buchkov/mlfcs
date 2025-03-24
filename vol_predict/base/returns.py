from __future__ import annotations

import numpy as np
import pandas as pd


def simple_to_log_returns(simple_returns: pd.DataFrame) -> pd.DataFrame:
    return np.log(1 + simple_returns)

def log_to_simple_returns(log_returns: pd.DataFrame) -> pd.DataFrame:
    return np.exp(log_returns) - 1


class Returns:
    def __init__(self, simple_returns: pd.DataFrame | None = None, log_returns: pd.DataFrame | None = None) -> None:
        self.simple_returns = simple_returns
        self.log_returns = log_returns

        if simple_returns is not None:
            self.log_returns = simple_to_log_returns(simple_returns)
        elif log_returns is not None:
            self.simple_returns = log_to_simple_returns(log_returns)

    def __call__(self, prices: pd.DataFrame) -> pd.DataFrame:
        self.prices = prices
        self.simple_returns = self.prices.pct_change()
        self.log_returns = simple_to_log_returns(self.simple_returns)
        return self.simple_returns

    def truncate(self, n_periods: int) -> Returns:
        return Returns(self.simple_returns.iloc[n_periods:], self.log_returns.iloc[n_periods:])
