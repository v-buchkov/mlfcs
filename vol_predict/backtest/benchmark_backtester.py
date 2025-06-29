import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler


class BacktestResults():
    def __init__(self,
                model,
                last_train_date: datetime = datetime.strptime("2018-07-01 00:00:00", "%Y-%m-%d %H:%M:%S"),
                is_training_expanded: bool = True,
                forecast_horizon: int = 1,
                forecasts: pd.Series = None,
                label_name: str = None,
                feature_names: list = None,
                true_vola: pd.Series = None):
        self.model = model
        self.model_name = model.name
        self.last_train_date = last_train_date
        self.is_training_expanded = is_training_expanded
        self.forecast_horizon = forecast_horizon
        self.label_name = label_name
        self.feature_names = feature_names
        self.forecasts = forecasts

        if forecasts.shape[0] != true_vola.shape[0]:
            raise ValueError("The number of rows in the forecasts and true volatility must be the same.")

        self.true_vola = true_vola
        self.residuals = (self.true_vola - forecasts) # nans for timestamps where forecasts are missing

        first_test_datetime = forecasts.index[0]
        squared_residuals = self.residuals.loc[first_test_datetime:]**2
        self.mse = squared_residuals.mean()
        self.rmse = np.sqrt(self.mse)
        absolute_residuals = np.abs(self.residuals.loc[first_test_datetime:])
        self.mae = absolute_residuals.mean()

        print(f"Backtest finished successfully.")
        print("---------------------------------------------------")
        print(f"Model: {self.model_name}")
        print(f"RMSE: {self.rmse:.7f}")
        print(f"MAE:  {self.mae :.7f}")
        #print(f"Weekly RMSE: {self.weekly_metrics_mean.rmse:.7f} +/- {self.weekly_metrics_std.rmse:.7f}")
        #print(f"Weekly MAE:  {self.weekly_metrics_mean.mae:.7f} +/- {self.weekly_metrics_std.mae:.7f}")
        print(f"Expanding training set: {self.is_training_expanded}")
        #print(f"Forecast horizon: {self.forecast_horizon}")
        print(f"Test starting date: {self.last_train_date}")    
        print("---------------------------------------------------")
        # print(f"Latex formatting:")
        # print(f"{self.model_name} &  "
        #     + f"{(1e3 * self.weekly_metrics_mean.rmse):.7f} &  "
        #     + f"{(1e3 * self.weekly_metrics_std.rmse):.7f}  &  "
        #     + f"{(1e3 * self.weekly_metrics_mean.mae):.7f}  &  "
        #     + f"{(1e3 *self.weekly_metrics_std.mae):.7f}  \\\\")

        plt.plot(forecasts, label=self.model_name)
        plt.plot(self.true_vola, label="Realized", alpha=0.5)
        plt.ylim(0, 0.001)
        plt.legend()
        plt.show()


class BenchmarkBacktester():
    """
    Takes a benchmark model, and performs an out-of-sample backtest, with periodic refitting of the 
    model during the backtest. Models should store the data internally. They should keep track of 
    potential scaling, and other preprocessing steps.

    Every model used as a benchmark must implement the following methods:
    - fit: fit the model to the data
    - forecast: forecast the next steps volatility
    - update: update the model with new observation without refitting.

    """
    def __init__(
        self,
        dataset: pd.DataFrame,
        output_template: pd.DataFrame,
        last_train_date: datetime = datetime.strptime("2018-06-30 23:59:59", "%Y-%m-%d %H:%M:%S"),
        is_training_expanded: bool = True,
        forecast_horizon: int = 1,
        lookback = pd.Timedelta(days=30),
        *args,
        **kwargs,
    ):
        """
        TODO add docstring
        """
        self.dataset = dataset
        self.output_template = output_template
        self.last_train_date = last_train_date
        self.first_test_datetime = output_template.index[0]
        self.is_training_expanded = is_training_expanded
        self.forecast_horizon = forecast_horizon
        self.lookback = lookback
        self.bias_lookback = pd.Timedelta(hours=24) # used to de-bias prediction if modelling log(y)


    def backtest(self, 
                 benchmark, 
                 hyperparams=None, 
                 use_ob_feats=True, 
                 use_log_y=True) -> BacktestResults:
        """
        TODO add docstring
        """

        # Instantiate the model. 
        if hyperparams is None:
            model = benchmark(use_ob_feats=use_ob_feats, use_log_y=use_log_y)
        else:
            model = benchmark(**hyperparams, use_ob_feats=use_ob_feats, use_log_y=use_log_y)
    
        if not hasattr(model, 'label_name'):
            model.label_name = 'log_vol' if use_log_y else 'vol'

        output = self.output_template.copy().rename(
            columns={"model_name": model.name}
        )

        for t in tqdm(output.index):
            # Check if we need to refit the model
            if output.loc[t, 'retraining_flag']:
                # forecast and update
                if self.is_training_expanded:
                    """ refit the model with all data up to t (excluding t)
                        t is monday 00:00:00, so before we forecast value for that t, we will refit
                        the model using data up to t but excluding t (thus t - pd.Timedelta(hours=1))

                        Then, forecasting is done using the data up to t (including t) but we reference
                        to the lagged data (shifted_dataset) to forecast the next steps. 
                        
                        So, X=dataset.loc[:t-pd.Timedelta(hours=1)] is two steps behind the
                        t, once because we are not yet at t, and the other because we are using
                        using features which are lagged once with respect to the label y.
                    """
                    model.fit(y=self.dataset.loc[:t-pd.Timedelta(hours=1), model.label_name],
                              X=self.dataset.loc[:t-pd.Timedelta(hours=1), model.feature_names])
                else:
                    # refit the model with only the data from t-lookback up to t
                    model.fit(y=self.dataset.loc[t-self.lookback:t-pd.Timedelta(hours=1), model.label_name],
                              X=self.dataset.loc[t-self.lookback:t-pd.Timedelta(hours=1), model.feature_names])

            pred = model.forecast(steps=self.forecast_horizon,
                                  X=self.dataset.loc[:t, model.feature_names])
            
            if use_log_y:
                conditional_log_vola = np.var(self.dataset.loc[t-self.bias_lookback:t-pd.Timedelta(hours=1), 'log_vol'].values)
                # cond_vola should be variance of residuals but the two are very similar because 
                # of relatively low predictability of volatility
                pred = pred * np.exp(conditional_log_vola/2)

            # TODO add support for multi-step forecasts
            output.loc[t, model.name] = pred[0]  # assuming single step forecast 

            model.update(new_y = self.dataset.loc[t:t, model.label_name],
                         new_X = self.dataset.loc[t:t, model.feature_names])

        # Return the forecasts
        return BacktestResults(model=model,
                               last_train_date=self.last_train_date,
                               is_training_expanded=self.is_training_expanded,
                               forecast_horizon=self.forecast_horizon,
                               forecasts=output.loc[:, model.name],
                               label_name=model.label_name,
                               feature_names=model.feature_names,
                               true_vola=self.dataset.loc[self.first_test_datetime:, 'vol'])


