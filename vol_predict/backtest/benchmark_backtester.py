import numpy as np
import pandas as pd

from typing import Union

from datetime import datetime

from tqdm import tqdm


class BacktestResults():
    def __init__(self,
                    model,
                    last_train_date: datetime = datetime.strptime("2018-07-01 00:00:00", "%Y-%m-%d %H:%M:%S"),
                    is_training_expanded: bool = True,
                    forecast_horizon: int = 1,
                    forecasts: pd.DataFrame = None,
                    true_vola: pd.Series = None):
        self.model = model
        self.model_name = model.__class__.__name__
        self.last_train_date = last_train_date
        self.is_training_expanded = is_training_expanded
        self.forecast_horizon = forecast_horizon

        if forecasts.shape[0] != true_vola.shape[0]:
            raise ValueError("The number of rows in the forecasts and true volatility must be the same.")
        if forecasts.shape[1] != forecast_horizon:
            raise ValueError("The number of columns in the forecasts must be equal to the forecast horizon.")
        
        if forecasts.shape[1] == 1:
            self.forecasts = forecasts.h1
        else:
            raise NotImplementedError("The model must return a single step forecast. Please set the forecast_horizon to 1.")
        
        self.true_vola = true_vola

        self.residuals = (self.true_vola - self.forecasts) # nans for timestamps where forecasts are missing

        first_test_datetime = forecasts.index[0]
        self.mse = (self.residuals.loc[first_test_datetime:]**2).mean()
        self.rmse = np.sqrt(self.mse)
        self.mae = (np.abs(self.residuals.loc[first_test_datetime:])).mean()
    
        print(f"Backtest finished successfully.")
        print("---------------------------------------------------")
        print(f"Model: {self.model_name}")
        print(f"RMSE: {self.rmse:.7f}")
        print(f"MAE:  {self.mae :.7f}")
        print(f"Expanding training set: {self.is_training_expanded}")
        print(f"Forecast horizon: {self.forecast_horizon}")
        print(f"Test starting date: {self.last_train_date}")    
        print("---------------------------------------------------")



class BenchmarkBacktester():
    """
    Takes a benchmark model, and performs an out-of-sample backtest, with periodic refitting of the 
    model during the backtest. Models should store the data internally. They should keep track of 
    potential scaling, and other preprocessing steps.

    Every model used as a benchmark must implement the following methods:
    - fit: fit the model to the data
    - forecast: forecast the next steps volatility
    - update: update the model with new observation without refitting.

    TODO maybe
    - get_params: get the parameters of the model
    - get_residuals: get the residuals of the model
    - summary: return a summary of the model (statsmodels style, or similar)

    Parameters
    ----------
    model: ModelWrap
        The model to be used for the backtest.

        Currently supported models are instances of:
        - `arch_model` from `arch` package
        - statsmodels.tsa.arima.model.ARIMA
        - statsmodels.tsa.statespace.sarimax.SARIMAX
        - our own EWMA model

    """
    def __init__(
        self,
        dataset: pd.DataFrame,
        last_train_date: datetime = datetime.strptime("2018-06-30 23:59:59", "%Y-%m-%d %H:%M:%S"),
        is_training_expanded: bool = True,
        forecast_horizon: int = 1,
        lookback = pd.Timedelta(weeks=4.5),
        *args,
        **kwargs,
    ):
        self.dataset = dataset.iloc[1:,:]
        self.shifted_dataset = dataset.copy().shift(1).dropna() 
        self.last_train_date = last_train_date
        self.first_test_datetime = dataset.loc[last_train_date:].index[1]
        self.is_training_expanded = is_training_expanded
        self.forecast_horizon = forecast_horizon

        if not is_training_expanded:
            self.lookback = lookback

        # every monday after the last training date
        self.refit_dates = [day for day in (pd.date_range(start=self.last_train_date, 
                                                          end=self.dataset.index[-1],
                                                          freq='D')
                                            ) if day.strftime("%A") == "Monday"]
        self.refit_dates += [self.refit_dates[-1] + pd.Timedelta(days=7)] # to include the last monday in the backtest
        self.test_timestamps = self.dataset.loc[self.first_test_datetime:].index


    def backtest(self, model_cls, hyperparams=None, is_multivariate=False) -> BacktestResults:
        """
        Use this method to fit daily. 

        Parameters
        ----------
        model : object
            The model class to be used for the backtest. It should implement the following methods:
            - fit: fit the model to the data
            - forecast: forecast the next steps volatility
            - update: update the model with new observation without refitting.
        """

        # Instantiate the model. 
        if hyperparams is None:
            if is_multivariate:
                model = model_cls(is_multivariate=is_multivariate)
            else:
                model = model_cls()
        else:
            model = model_cls(**hyperparams, is_multivariate=is_multivariate)


        # Fit the model to the initial training data
        model.fit(y=self.dataset.loc[:self.last_train_date, 'vol'],
                  X=self.shifted_dataset.loc[:self.last_train_date, model.feature_names])

        # initialize monday counters
        prev_monday = self.refit_dates[0] - pd.Timedelta(days=7)
        next_monday = self.refit_dates[0]

        # create an empty dataframe to store the forecasts
        forecasts = pd.DataFrame(index = self.test_timestamps,
                                 columns = [f'h{h}' for h in range(1, self.forecast_horizon + 1)])

        for t in tqdm(self.test_timestamps):
            # Check if we need to refit the model
            if prev_monday < t and t >= next_monday:
                # Refit the model when you reach the next monday
                prev_monday = next_monday
                next_monday = self.refit_dates[self.refit_dates.index(next_monday) + 1]
                # forecast and update
                if self.is_training_expanded:
                    """ refit the model with all data up to t (excluding t)
                        t is monday 00:00:00, so before we forecast value for that t, we will refit
                        the model using data up to t but excluding t (thus t - pd.Timedelta(hours=1))

                        Then, forecasting is done using the data up to t (including t) but we reference
                        to the lagged data (shifted_dataset) to forecast the next steps. 
                        
                        So, X=shifted_dataset.loc[:t-pd.Timedelta(hours=1)] is two steps behind the
                        t, once because we are not yet at t, and the other because we are using
                        using features which are lagged once with respect to the label y.
                    """
                    model.fit(y=self.dataset.loc[:t-pd.Timedelta(hours=1), 'vol'],
                              X=self.shifted_dataset.loc[:t-pd.Timedelta(hours=1), model.feature_names])
                else:
                    # refit the model with only the data from t-lookback up to t
                    model.fit(y=self.dataset.loc[t-lookback:t-pd.Timedelta(hours=1), 'vol'],
                              X=self.shifted_dataset.loc[t-lookback:t-pd.Timedelta(hours=1), model.feature_names])

            
            forecasts.loc[t] = model.forecast(steps=self.forecast_horizon,
                                              X=self.shifted_dataset.loc[:t, model.feature_names])
            
            model.update(new_y = self.dataset.loc[t:t, 'vol'],
                         new_X = self.dataset.loc[t:t, model.feature_names])


        # Return the forecasts
        return BacktestResults(model=model,
                               last_train_date=self.last_train_date,
                               is_training_expanded=self.is_training_expanded,
                               forecast_horizon=self.forecast_horizon,
                               forecasts=forecasts,
                               true_vola=self.dataset.loc[self.first_test_datetime:, 'vol'])


