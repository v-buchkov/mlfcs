# abstract class for model wrapper
from __future__ import annotations
from typing import Union
import numpy as np
import pandas as pd

from abc import ABC, abstractmethod


class AbstractBenchmark(ABC):
    """
    An alternative to `AbstractPredictor` class. It is used to wrap models from
    `arch` and `statsmodels` packages, as well as custom made benchmarks,
    which are not PyTorch models and require a different training
    and inference process (storing residuals, etc.).
    """

    def __init__(self):
        pass

    @abstractmethod
    def fit(self, y: pd.Series, X: pd.DataFrame):
        """
        Fit the model to the data.

        Parameters
        ----------
        y : pd.Series
            The volatility to fit the model to.
        X : pd.DataFrame
            The volatility and order book features to fit the model to.
            THEY ARE ONCE LAGGED WITH RESPECT TO y.
        """

    @abstractmethod
    def forecast(self, steps: int = 1, X: pd.DataFrame = None) -> np.ndarray:
        """
        Forecast the model for a given number of steps.

        Parameters
        ----------
        steps : int
            The number of steps to forecast.
        X : pd.DataFrame
            The volatility and order book features to forecast the model with.
        """

    @abstractmethod
    def update(self, new_y: pd.Series, new_X: pd.DataFrame):
        """
        Update the model with new data.

        Parameters
        ----------
        new_y : pd.Series
            The new volatility to update the model with.
        new_X : pd.DataFrame
            The new volatility and order book features to update the model with.
            X is NOT LAGGED with respect to y here, because update is what happens after we observe
            label y and features X.
        """
