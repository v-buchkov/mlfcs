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
    def _fit(self, 
            y: pd.Series,
            X: pd.DataFrame):
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
    def _forecast(self, steps: int=1, X: pd.DataFrame=None) -> np.ndarray:
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
    def _update(self, 
               new_y: pd.Series,
               new_X: pd.DataFrame):
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


    # def _compute_pc(self, X_train: pd.DataFrame, n_components: int = 5) -> pd.DataFrame:
    #     self.scaler = StandardScaler()
    #     X_train_scaled = self.scaler.fit_transform(X_train)

    #     self.pca = PCA(n_components=n_components)
    #     pcs = self.pca.fit_transform(X_train_scaled)

    #     return pd.DataFrame(data=pcs, columns=[f"pc_{i+1}" for i in range(n_components)], index=X_train.index)

    def _log_transform(self, y: pd.Series) -> pd.Series:
        return pd.Series(np.log(y), index=y.index, name=y.name)

    def fit(self,
            y: pd.Series,
            X: pd.DataFrame):
        """
        Fit the model to the data.

        Parameters
        ----------
        y : pd.Series
            The volatility to fit the model to.
        X : pd.DataFrame
            The volatility and order book features to fit the model to.
            THEY ARE ONCE LAGGED WITH RESPECT TO y.
            If X contains 'vol' it means the model is autoregressive 
        """
        if self.use_log_y:
            # If log transformation is used, transform y
            y = self._log_transform(y)
        self._fit(y=y, X=X)

    def forecast(self, steps: int=1, X: pd.DataFrame=None) -> np.ndarray:
        """
        Forecast the model for a given number of steps.

        Parameters
        ----------
        steps : int
            The number of steps to forecast.
        X : pd.DataFrame
            The volatility and order book features to forecast the model with.
            For autoregressive models X contains 'vol'.
        
        Returns
        -------
        np.ndarray or pd.Series
            The forecasted values.
        """
        # if self.do_pca:
        #     self.ob_feature_names = ...
        #     # If PCA is enabled, compute PCA components
        #     X = self._compute_pc(X.loc[:, self.ob_feature_names], n_components=self.pca_n_components)
        return self._forecast(steps=steps, X=X)

    def update(self, 
              new_y: pd.Series,
              new_X: pd.DataFrame):
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
        if self.use_log_y:
            # If log transformation is used, transform new_y
            new_y = self._log_transform(new_y)
        self._update(new_y=new_y, new_X=new_X)


    # TODO rename `forecast` to `_forecast` and make new `forecast` method for the AbstractBenchmark
    # which is not an abstract method, but rather a wrapper for the `_forecast` method and adds bias
    # correction for log transformed volatility, or implements more general calibration methods.