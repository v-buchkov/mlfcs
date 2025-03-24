from enum import Enum

import pandas as pd

from vol_predict.base.returns import Returns


class VolatilityMethod(Enum):
    SAMPLE_VOL = "sample_vol"
    SQUARED_RETURNS = "squared_returns"


class VolatilityCalculator:
    def __init__(self, method: VolatilityMethod):
        self.method = method

    def __call__(self, returns: pd.DataFrame) -> pd.DataFrame:
        if self.method == VolatilityMethod.SAMPLE_VOL:
            return returns.std()
        elif self.method == VolatilityMethod.SQUARED_RETURNS:
            return returns.pow(2).mean().pow(0.5)
        else:
            raise ValueError(f"Unknown volatility method: {self.method}")


class Volatility:
    def __init__(
            self,
            returns: Returns | None = None,
            volatilities: pd.DataFrame | None = None,
            method: str | None = None,
    ):
        assert returns is not None or volatilities is not None, "Either returns or volatilities must be provided!"

        assert returns is None or volatilities is None, "Only one of returns or volatilities must be provided!"

        self.returns = returns.log_returns

        self.method = VolatilityMethod(method) if method is not None else VolatilityMethod.SAMPLE_VOL
        self.volatilities = self._calculate_volatilities() if volatilities is None else volatilities

    def _calculate_volatilities(self) -> pd.DataFrame:
        volatility_calculator = VolatilityCalculator(self.method)
        return volatility_calculator(self.returns)

    def __call__(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        return self.volatilities
