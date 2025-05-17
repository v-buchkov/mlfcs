from vol_predict.models.abstract_benchmark import AbstractBenchmark
import numpy as np
import pandas as pd


import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
import time
import math
import matplotlib.pyplot as plt

sts = tfp.sts
mcmc = tfp.mcmc
tfd = tfp.distributions
tfb = tfp.bijectors
vi = tfp.vi

# from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_squared_error
# from sklearn.preprocessing import StandardScaler

from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from arch.univariate import HARX


class Naive(AbstractBenchmark):
    """
    Naive predictor. Given the past volatility, predict the next step as the last observed value.
    """

    def __init__(
        self,
    ):
        self.feature_names = ['vol']
        self.last_vol = None
    
    
    def fit(self, 
            y: np.ndarray,
            X: np.ndarray):
        self.last_vol = X.vol.values[-1] # not used, since forecast will always receive X


    def forecast(self, steps: int=1, X: np.ndarray=None) -> np.ndarray:
        if steps == 1:
            return X.vol.values[-1]
        else:
            # If we want to forecast more than one step, we need to return the last observed value
            # for each step. This is a naive predictor, so we just return the same value.
            return np.full(steps, X.vol.values[-1])


    def update(self, 
               new_y: np.ndarray,
               new_X: np.ndarray):
        self.last_observed = new_X.vol.values[-1]


class EWMA(AbstractBenchmark):
    """
    Same logic as `EWMAPredictor`, but with an interface for BenchmarkBacktester.

    Given the past volatility, average them by weighting the most recent ones more heavily. 
    The forgetting factor and half-life are related as follows:

    forgetting_factor = 0.5 ** (1 / half_life)

    """

    def __init__(
        self,
        look_back: int,
        half_life: float,
        *args,
        **kwargs,
    ):
        assert look_back > 0.0, "ewma_look_back_win must be greater than 0"
        assert half_life > 0.0,     "ewma_half_life must be greater than 0"
        self.look_back = look_back
        self.half_life = half_life
        self.hyperparams = {
            "look_back": look_back,
            "half_life": half_life,
        }
        self.feature_names = ['vol']

        self.forgetting_factor = self._hl_to_ff(self.half_life)
        self.exp_weights = np.zeros(self.look_back)
        for i in range(self.look_back):
            self.exp_weights[i] = self.forgetting_factor ** (i)

        self.normalization_const = (1 - self.forgetting_factor) / (
            1 - self.forgetting_factor**self.look_back
        )

        self.stored_data = None

        # # forgetting_factor is bounded between 0 and 1
        # # if needed, raw_forgetting_factor is an unbounded version of forgetting_factor
        # # sigmoid_inverse(3.4) = 0.9677, which is a forgetting_factor equivalent to half-life of 21
        # self.raw_forgetting_factor = Parameter(torch.Tensor([3.4]), requires_grad=True)
        # self.forgetting_factor = torch.sigmoid(self.raw_forgetting_factor)

    def _hl_to_ff(self, half_life: float) -> float:
        """
        Convert half-life to forgetting factor.
        """
        assert half_life > 0.0, "half_life must be greater than 0"
        return 0.5 ** (1 / half_life)

    def _ff_to_hl(self, forgetting_factor: float) -> float:
        """
        Convert forgetting factor to half-life.
        """
        assert forgetting_factor > 0.0, (
            "forgetting_factor must be greater than 0"
        )
        assert forgetting_factor < 1.0, "forgetting_factor must be less than 1"
        return -np.log(2.0) / np.log(forgetting_factor)

    def fit(self,
            y: pd.Series,
            X: pd.DataFrame):
        """
        There is nothing to be fitted, except to store the last calculation_win volatilties

        """
        # vol = X.vol.values
        # calculation_win = min(self.look_back, vol.shape[0])
        # if calculation_win == 0:
        #     raise ValueError("Empty past_data or zero look_back resulted in calculation_win=0")
        # elif calculation_win == self.look_back:
        #     self.stored_data = vol[-self.look_back:]
        # elif calculation_win < self.look_back:
        #     # if we don't have enough data, we need to normalize the weights to a smaller window
        #     # normalization_const = (1 - self.forgetting_factor) / (
        #     #     1 - self.forgetting_factor**calculation_win
        #     # )
        #     # # reweight
        #     # self.exp_weights[:calculation_win] *= normalization_const
        #     self.stored_data = vol[-calculation_win:]
        # else:
        #     raise ValueError(
        #         f"calculation_win={calculation_win}"
        #     )

        # TODO: Implement fitting of forgetting factor (i.e. half-life).
        # For now, we just use the forgetting factor passed in the constructor.


    def forecast(
        self,
        steps: int = 1,
        X: pd.DataFrame=None) -> float:

        # past_vol = X.vol.values
        # calculation_win = min(self.look_back, self.stored_data.shape[0])
        # if calculation_win == 0:
        #     return None
        # elif calculation_win == self.look_back:
        #     return self.normalization_const * np.sum(self.exp_weights * self.stored_data[-calculation_win:])
        # elif calculation_win < self.look_back:
        #     # if we don't have enough data, we need to normalize the weights to a smaller window
        #     normalization_const = (1 - self.forgetting_factor) / (
        #         1 - self.forgetting_factor**calculation_win
        #     )
        #     return normalization_const * np.sum(
        #         self.exp_weights[:calculation_win] * self.stored_data[-calculation_win:]
        #     )
        # else:
        #     raise ValueError(
        #         f"calculation_win={calculation_win}"
        #     )
    
        past_vol = X.vol.values
        calculation_win = min(self.look_back, past_vol.shape[0])
        if calculation_win == 0:
            return None
        elif calculation_win == self.look_back:
            return self.normalization_const * np.sum(self.exp_weights * past_vol[-calculation_win:])
        elif calculation_win < self.look_back:
            # if we don't have enough data, we need to normalize the weights to a smaller window
            normalization_const = (1 - self.forgetting_factor) / (
                1 - self.forgetting_factor**calculation_win
            )
            return normalization_const * np.sum(
                self.exp_weights[:calculation_win] * past_vol[-calculation_win:]
            )
        else:
            raise ValueError(
                f"calculation_win={calculation_win}"
            )
    
    def update(self, new_y: np.ndarray, new_X: np.ndarray):
        """
        Update the model with new data.
        """
        # if self.stored_data.shape[0] == self.look_back:
        #     # remove the oldest data
        #     self.stored_data = np.roll(self.stored_data, -1, axis=0)
        #     self.stored_data[-1] = new_data
        # else:
        #     # add the new data
        # self.stored_data = np.append(self.stored_data, new_y.vol.values)


class ARIMAX(AbstractBenchmark):
    """
    ARIMAX volatility predictor (we know already that vola is stationary). 
    Given the past volatility, predict the next one using ARIMA model.
    """
    def __init__(
        self,
        p: int,
        d: int,
        q: int,
        is_multivariate: bool = False,
        *args,
        **kwargs,
    ):
        self.p = p
        self.d = d
        self.q = q
        self.order = (p, d, q)
        self.is_multivariate = is_multivariate

        if is_multivariate:
            # TODO properly name ob_features
            self.feature_names = ['mean_spread', 'mean_weighted_spread']
        else:
            self.feature_names = []

        self.results = None

    def fit(self, y: pd.Series, X: pd.DataFrame):
        """

        """

        if self.is_multivariate:
            self.model = ARIMA(endog=y.values, exog=X.values, order=self.order) 
            self.stored_y = y.values
            self.stored_X = X.values  
        else:
            self.model = ARIMA(endog=y.values, order=self.order)
            self.stored_y = y.values

        if self.results is None:
            self.results = self.model.fit()
            self.residuals = self.results.resid
        else:
            self.results = self.model.fit(
                start_params=self.results.params, # should help with convergence 
            )
            self.residuals = self.results.resid

    def forecast(self, steps: int = 1, X: pd.DataFrame = None):
        """
        Forecast the next steps volatility.
        """
        if self.results is None:
            raise ValueError("Model not fitted yet. Please fit the model first.")

        if self.is_multivariate:
            return self.results.forecast(steps=steps, exog=X.values[-1])
        else:
            return self.results.forecast(steps=steps)

    def update(self, 
               new_y: pd.Series,
               new_X: pd.DataFrame):
        """
        Update the model with new data.
        new_data: numpy array of new observations. The new data is not used to refit the model, but
        to update the fitted ARIMA ... 
        """
        self.stored_y = np.append(self.stored_y, new_y)
        if self.is_multivariate:
            self.stored_X = np.append(self.stored_X, new_X.values, axis=0)
            self.results = self.results.apply(endog=self.stored_y, exog=self.stored_X, refit=False)
        else:
            self.results = self.results.apply(endog=self.stored_y, refit=False)
        self.residuals = self.results.resid


class GARCH():
    """
    GARCH volatility predictor (we know already that vola is stationary). 
    Given the past volatility, predict the next one using GARCH model.
    """
    def __init__(
        self,
        p: int,
        o: int,
        q: int,
        dist: str = 'studentst',
        scale: float = 1.0,
        is_multivariate: bool = False,
        type: str = 'Garch',
        *args,
        **kwargs,
    ):
        self.p = p
        self.o = o
        self.q = q
        self.order = (p, o, q)
        self.dist = dist
        self.scale = scale
        self.is_multivariate = is_multivariate
        self.type = type

        if is_multivariate:
            # TODO properly name ob_features
            self.feature_names = ['mean_spread', 'mean_weighted_spread']
        else:
            self.feature_names = []

        self.results = None

        self.label_name = 'ret'
        # #self.residuals = None
        # if is_multivariate:
        #     # TODO properly name ob_features
        #     self.feature_names = ['ret', 'TO_BE_IMPLEMENTED']
        # else:
        #     self.feature_names = ['ret']
        

    def fit(self, y: pd.Series, X: pd.DataFrame):
        """
        """

        if self.is_multivariate:
            self.stored_y = y.values
            self.stored_X = X.values
            self.model = arch_model(
                y=self.stored_y, 
                x=self.stored_X, 
                vol=self.type,
                p=self.p,
                o=self.o,
                q=self.q,
                dist=self.dist,
                #rescale=True,
                )
  
        else:
            self.stored_y = y.values
            self.model = arch_model(
                y=self.stored_y, 
                vol=self.type,
                p=self.p,
                o=self.o,
                q=self.q,
                dist=self.dist,
                #rescale=True,
                )

        # If there is not start_params use this
        self.results = self.model.fit(disp="off")

        # if self.results is None:
        #     self.results = self.model.fit(disp="off")
        #     #self.results = self.model.fit()
        #     #self.residuals = self.results.resid
        # else:
        #     self.results = self.model.fit(
        #         start_params=self.results.params, # should help with convergence 
        #     )
        #     #self.residuals = self.results.resid



    def forecast(self, steps: int = 1, X: pd.DataFrame = None):
        """
        Forecast the next steps volatility.
        """
        if self.results is None:
            raise ValueError("Model not fitted yet. Please fit the model first.")

        if self.is_multivariate:
            return self.results.forecast(horizon=steps).variance.values[0][0]
            # return self.results.conditional_volatility[-1] FIXME
        else:
            return self.results.forecast(horizon=steps).variance.values[0][0]
            # return self.results.conditional_volatility[-1] FIXME


    def update(self, new_y: pd.Series, new_X: pd.DataFrame):
        """
        Update the model with new data.
        """
        self.stored_y = np.append(self.stored_y, new_y.values)
        if self.is_multivariate:
            self.stored_X = np.append(self.stored_X, new_X.values, axis=0)
            self.model = arch_model(
                y=self.stored_y, 
                x=self.stored_X, 
                vol=self.type,
                p=self.p,
                o=self.o,
                q=self.q,
                dist=self.dist,
                #rescale=True,
                )
        else:
            self.model = arch_model(
                y=self.stored_y , 
                vol=self.type,
                p=self.p,
                o=self.o,
                q=self.q,
                dist=self.dist,
                #rescale=True,
                )
        
        self.results = self.model.fix(self.results.params.values)


class HAR():
    """
    HAR volatility predictor (we know already that vola is stationary). 
    Given the past volatility, predict the next one using GARCH model.
    """
    def __init__(
        self,
        lags: list,
        distribution: str = 'studentst',
        is_multivariate: bool = False,
        *args,
        **kwargs,
    ):
        self.lags = lags
        self.distribution = distribution
        self.is_multivariate = is_multivariate

        if is_multivariate:
            # TODO properly name ob_features
            self.feature_names = ['mean_spread', 'mean_weighted_spread']
        else:
            self.feature_names = []

        self.results = None

        self.label_name = 'vol'
        

    def fit(self, y: pd.Series, X: pd.DataFrame):
        """
        """

        if self.is_multivariate:
            self.stored_y = y.values
            self.stored_X = X.values
            self.model = HARX(
                y=self.stored_y, 
                x=self.stored_X, 
                lags=self.lags,
                distribution=self.distribution,
                #rescale=True,
                )
  
        else:
            self.stored_y = y.values
            self.model = HARX(
                y=self.stored_y, 
                lags=self.lags,
                distribution=self.distribution,
                #rescale=True,
                )

        # If there is not start_params use this
        self.results = self.model.fit(disp="off")


    def forecast(self, steps: int = 1, X: pd.DataFrame = None):
        """
        Forecast the next steps volatility.
        """
        if self.results is None:
            raise ValueError("Model not fitted yet. Please fit the model first.")

        if self.is_multivariate:
            return self.results.forecast(horizon=steps).mean.values[0][0]
        else:
            return self.results.forecast(horizon=steps).mean.values[0][0]


    def update(self, new_y: pd.Series, new_X: pd.DataFrame):
        """
        Update the model with new data.
        """
        self.stored_y = np.append(self.stored_y, new_y.values)
        if self.is_multivariate:
            self.stored_X = np.append(self.stored_X, new_X.values, axis=0)
            self.model = HARX(
                y=self.stored_y, 
                x=self.stored_X, 
                lags=self.lags,
                distribution=self.distribution,
                #rescale=True,
                )
        else:
            self.model = HARX(
                y=self.stored_y, 
                lags=self.lags,
                distribution=self.distribution,
                #rescale=True,
                )
        
        self.results = self.model.fix(self.results.params.values)



# RF
# XGBoost
# Enet





class STS(AbstractBenchmark):
    def __init__(self,
                 level=True,
                 trend=False,
                 season=False,
                 seasonal_period=0, 
                 learning_rate=1e-4,
                 num_steps=int(1e4), 
                 sparse=True,
                 is_multivariate=True):


        self.level = level
        self.trend = trend
        self.season = season
        self.seasonal_period = seasonal_period
        self.lr = learning_rate
        self.num_steps = num_steps
        self.sparse = sparse
        self.preds = []
        if is_multivariate:
            self.feature_names = ['vol', 'mean_spread', "mean_weighted_spread"]
        else:
            self.feature_names = ['vol']
        self.surrogate_posterior = None
    
    # Helper function that builds a tfp.sts model. Returns both the model components and the built model itself
    def _build_model(self, 
                    y_train, 
                    X_train=None,
                    level=True,
                    trend=False,
                    season=False,
                    seasonal_period=0,
                    sparse=True):
        components = []
        if (level==True and trend==False):
            level_component = sts.LocalLevel(observed_time_series=y_train, name="level")
            components.append(level_component)
        if (level==True and trend==True):
            local_linear_trend_component = sts.LocalLinearTrend(observed_time_series=y_train, name="trend")
            components.append(local_linear_trend_component)
        if season==True:
            season_component = sts.Seasonal(num_seasons=seasonal_period,
                                            observed_time_series=y_train,
                                            name="seasonality")
            components.append(season_component)
        if X_train is not None:
            if sparse == True:
                order_book_feat = sts.SparseLinearRegression(design_matrix=X_train,
                                                             name="order_book_features",
                                                             )
            else:
                order_book_feat = sts.LinearRegression(design_matrix=X_train,
                                                       name="order_book_features")
            components.append(order_book_feat)
        
        model = sts.Sum(components=components, 
                             observed_time_series=y_train, 
                             name="sts_model")
        return components, model
    

    # This function returns a predictive distribution over future observations for of length horizon. First, we sample
    # the parameters of our model posterior_samples-times from the fitted surrogate posterior. Then we use the function
    # tfp.sts.forecast to produce predictive distribution over future observations for of length horizon.
    def _forecast_dist(self, posterior_samples=int(1e3), horizon=1):
        model_q_samples = self.surrogate_posterior.sample(posterior_samples)
        forecast_dist = tfp.sts.forecast(self.model,
                                         observed_time_series=self.y_train,
                                         parameter_samples=model_q_samples,
                                         num_steps_forecast=horizon)    
        return forecast_dist


    def fit(self, y: pd.Series, X: pd.DataFrame):
        y_train = y.values 
        X_train = X.values

        y = np.asarray(y_train, dtype=np.float32)
        self.y_mean = y.mean()
        self.y_std = y.std() if y.std() != 0 else 1.0
        y_stdized = (y - self.y_mean) / self.y_std
        
        if X_train is not None:
            X = np.asarray(X_train, dtype=np.float32)
            self.X_mean = X.mean(axis=0)
            self.X_std = X.std(axis=0)
            self.X_std[self.X_std == 0] = 1.0
            X_stdized = (X - self.X_mean) / self.X_std
        else:
            self.X_train = None
            self.X_mean = None
            self.X_std = None
        
        if self.surrogate_posterior is not None:
            self.surrogate_posterior = None

        self._fit(y_stdized, X_stdized)

        print(type(self.elbo_loss_history))
        plt.plot(self.elbo_loss_history)
        plt.title("ELBO loss")
        plt.xlabel("Iteration")
        plt.ylabel("ELBO loss")
        plt.show()

    # This function takes an univariate time series dataset y and fits our structural time series to the given data y 
    # via variational inference. By defautl, we use a learning rate of 1e-3 and 1e5 iterations. The function returns
    # the ELBO loss recorded during training and returns it.
    # @tf.function(reduce_retracing=True)
    def _fit(self, y_stdized, X_stdized):
        self.X_train = tf.convert_to_tensor(X_stdized, dtype=tf.float32)
        self.y_train = tf.convert_to_tensor(y_stdized, dtype=tf.float32)

        self.components, self.model = self._build_model(self.y_train,
                                                       self.X_train,
                                                       self.level,
                                                       self.trend,
                                                       self.season,
                                                       self.seasonal_period,
                                                       self.sparse)
        
        # We now build a surrogate posterior that factors over the model parameters

        self.surrogate_posterior = sts.build_factored_surrogate_posterior(self.model)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.optimizer.learning_rate = self.lr
        self.elbo_loss_history = tfp.vi.fit_surrogate_posterior(
            target_log_prob_fn=self.model.joint_distribution(observed_time_series=self.y_train).log_prob,
            surrogate_posterior=self.surrogate_posterior,
            optimizer=self.optimizer,
            num_steps=self.num_steps,
            jit_compile=True)


            
        
    def forecast(self, steps: int=1, X=None):
        dist = self._forecast_dist(posterior_samples=int(1e3), horizon=steps)

        m_std = dist.mean().numpy()[0,0]
        s_std = dist.stddev().numpy()[0,0]

        # back to original scale
        m = m_std * self.y_std + self.y_mean
        s = s_std * self.y_std
        preds.append({"y_pred_mean": m, "y_pred_std": s})

        return np.exp(m+0.5*s**2)
        
    # Builds a new tfp.sts model with the newly observed feature variables X_new and updates both self.components and 
    # self.model
    
    def _update_design_matrix(self, X_new):
        # 1) turn X_new into a (1, K) array and standardize it
        x = np.asarray(X_new, dtype=np.float32)
        x_std = (x - self.X_mean) / self.X_std
        x_t = tf.convert_to_tensor(x_std)
        if x_t.ndim == 1:
            x_t = tf.expand_dims(x_t, axis=0)
    
        print(f"X_train shape: {self.X_train.shape}")
        # 2) append only the standardized row
        self.X_train = tf.concat([self.X_train, x_t], axis=0)
        print(f"X_train shape after concat: {self.X_train.shape}")
    
        # 3) rebuild the model structure
        self.components, self.model = self._build_model(self.y_train,
                                                       self.X_train,
                                                       self.level,
                                                       self.trend,
                                                       self.season,
                                                       self.seasonal_period,
                                                       self.sparse)
        
    # Rebuilds the tfp.sts model with the newly observed target variable y_new and updates both self.components and 
    # self.model
    def _update_observed_time_series(self, y_new):
        #!!! Double check the standardizing
        y = np.asarray(y_new, dtype=np.float32)
        y_std = (y - self.y_mean) / self.y_std
        y_t = tf.convert_to_tensor([y_std], dtype=tf.float32)
        self.y_train = tf.concat([self.y_train, y_t], axis=0)
        
        #y_new = tf.convert_to_tensor([y_new], dtype=tf.float32)
        #observed_time_series_new = tf.concat([self.y_train, y_new], axis=0)
        #self.y_train = observed_time_series_new
        self.components, self.model = self._build_model(y_train=self.y_train,
                                                       X_train=self.X_train,
                                                       level=self.level,
                                                       trend=self.trend,
                                                       season=self.season,
                                                       seasonal_period=self.seasonal_period,
                                                       sparse=self.sparse)
    

    def update(self, new_y: pd.Series, new_X: pd.DataFrame):
        """
        Update the model with new data.
        """
        
        if new_X is not None:
            self._update_design_matrix(X_new=new_X.values)
        else:
            pass
        self._update_observed_time_series(y_new=new_y.values)
        





