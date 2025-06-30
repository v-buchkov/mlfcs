from vol_predict.models.abstract_benchmark import AbstractBenchmark
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import statsmodels.api as sm

from scipy.optimize import lsq_linear
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


class Naive(AbstractBenchmark):
    """
    Given the past volatility, predict the next step as the last observed value.
    """

    def __init__(
        self,
        use_ob_feats: bool = False,
        use_log_y: bool = False,
    ):
        self.feature_names = ["vol_lag1"]
        self.last_vol = None
        self.use_ob_feats = use_ob_feats
        self.use_log_y = use_log_y
        self.name = "Naive"

    def fit(self, y: pd.Series, X: pd.DataFrame):
        pass

    def forecast(self, steps: int = 1, X: pd.DataFrame = None) -> np.ndarray:
        if steps == 1:
            return X.loc[:, self.feature_names].values[-1]
        else:
            # If we want to forecast more than one step, we need to return the last observed value
            # `steps` times.
            return np.full(steps, X.loc[:, self.feature_names].values[-1])

    def update(self, new_y: pd.Series, new_X: pd.DataFrame):
        pass


class EWMA(AbstractBenchmark):
    """

    The forgetting factor and half-life are related as follows:

    forgetting_factor = 0.5 ** (1 / half_life)

    """

    def __init__(
        self,
        look_back: int,
        half_life: float,
        use_ob_feats: bool = False,
        use_log_y: bool = False,
        *args,
        **kwargs,
    ):
        self.use_log_y = use_log_y
        assert look_back > 0.0, "ewma_look_back_win must be greater than 0"
        assert half_life > 0.0, "ewma_half_life must be greater than 0"
        self.look_back = look_back
        self.half_life = half_life
        self.hyperparams = {
            "look_back": look_back,
            "half_life": half_life,
        }
        self.use_ob_feats = kwargs.pop("use_ob_feats", False)
        if self.use_log_y:
            self.label_name = "log_vol"
            self.feature_names = ["log_vol_lag1"]
        else:
            self.label_name = "vol"
            self.feature_names = ["vol_lag1"]

        self.name = "EWMA"

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
        assert forgetting_factor > 0.0, "forgetting_factor must be greater than 0"
        assert forgetting_factor < 1.0, "forgetting_factor must be less than 1"
        return -np.log(2.0) / np.log(forgetting_factor)

    def fit(self, y: pd.Series, X: pd.DataFrame):
        """
        There is nothing to be fitted, except to store the last calculation_win volatilties
        because forecast will receive all available data.

        # TODO: Implement fitting of forgetting factor (i.e. half-life).
        # For now, we just use the forgetting factor passed in the constructor.

        """
        pass

    def forecast(self, steps: int = 1, X: pd.DataFrame = None) -> float:
        past_vol = X.loc[:, self.feature_names].values.flatten()
        calculation_win = min(self.look_back, past_vol.shape[0])
        if calculation_win == 0:
            ret = None
        elif calculation_win == self.look_back:
            ret = self.normalization_const * np.sum(
                self.exp_weights * past_vol[-calculation_win:]
            )
        elif calculation_win < self.look_back:
            # if we don't have enough data, we need to normalize the weights to a smaller window
            normalization_const = (1 - self.forgetting_factor) / (
                1 - self.forgetting_factor**calculation_win
            )
            ret = normalization_const * np.sum(
                self.exp_weights[:calculation_win] * past_vol[-calculation_win:]
            )
        else:
            raise ValueError(f"calculation_win={calculation_win}")

        ret = np.array([ret])
        return ret

    def update(self, new_y: np.ndarray, new_X: np.ndarray):
        """
        Update the model with new data.
        """
        pass


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
        use_ob_feats: bool = False,
        use_log_y: bool = True,
        *args,
        **kwargs,
    ):
        self.p = p
        self.d = d
        self.q = q
        self.order = (p, d, q)
        self.use_ob_feats = use_ob_feats
        self.use_log_y = use_log_y
        self.label_name = "log_vol" if use_log_y else "vol"
        # self.bais_term_window = 24*5 # ~ 1 week of data

        self.name = "ARIMAX" if use_ob_feats else "ARIMA"
        # Models from `statsmodels` and arch are wrapped without vol_lag1 as feature
        # Models from `sklearn`, and custom made models are wrapped with vol_lag1 as feature
        # self.feature_names = ['vol_lag1']
        self.feature_names = []
        if use_ob_feats:
            self.feature_names += [
                "log_volume",
                "mean_spread",
                "mean_bid_depth",
                "mean_ask_volume",
            ]
        self.results = None

    def fit(self, y: pd.Series, X: pd.DataFrame):
        """ """

        if self.use_ob_feats:
            self.model = ARIMA(endog=y.values, exog=X.values, order=self.order)
            self.stored_y = y.values
            self.stored_X = X.values
        else:
            self.model = ARIMA(endog=y.values, order=self.order)
            self.stored_y = y.values

        self.results = self.model.fit()
        self.residuals = self.results.resid

        print(self.results.summary())

    def forecast(self, steps: int = 1, X: pd.DataFrame = None):
        """
        Forecast the next steps volatility.
        """
        if self.results is None:
            raise ValueError("Model not fitted yet. Please fit the model first.")

        if self.use_ob_feats:
            pred = self.results.forecast(steps=steps, exog=X.values[-1])
        else:
            pred = self.results.forecast(steps=steps)

        if self.use_log_y:
            # Bias correction is done by backtester
            return np.exp(pred)
        else:
            return pred

    def update(self, new_y: pd.Series, new_X: pd.DataFrame):
        """
        Update the model with new data.
        new_data: numpy array of new observations. The new data is not used to refit the model, but
        to update the fitted ARIMA ...
        """
        self.stored_y = np.append(self.stored_y, new_y)
        if self.use_ob_feats:
            self.stored_X = np.append(self.stored_X, new_X.values, axis=0)
            self.results = self.results.apply(
                endog=self.stored_y, exog=self.stored_X, refit=False
            )
            self.residuals = self.results.resid
        else:
            self.results = self.results.apply(endog=self.stored_y, refit=False)
            self.residuals = self.results.resid


class GARCH(AbstractBenchmark):
    """
    GARCH volatility predictor (we know already that vola is stationary).
    Given the past volatility, predict the next one using GARCH model.
    """

    def __init__(
        self,
        p: int,
        o: int,
        q: int,
        dist: str = "studentst",
        scale: float = 1.0,
        use_ob_feats: bool = False,
        use_log_y: bool = True,
        type: str = "Garch",
        *args,
        **kwargs,
    ):
        self.p = p
        self.o = o
        self.q = q
        self.order = (p, o, q)
        self.dist = dist
        self.scale = scale  # NOT used
        self.use_ob_feats = use_ob_feats
        self.use_log_y = use_log_y
        self.type = type
        self.name = self.type
        if use_ob_feats:
            self.name += "X"
        self.name += f"_{dist}"

        # Models from `statsmodels` and arch are wrapped without vol_lag1 as feature
        # Models from `sklearn`, and custom made models are wrapped with vol_lag1 as feature
        # self.feature_names = ['vol_lag1']
        self.feature_names = []
        if use_ob_feats:
            self.feature_names += [
                "mean_spread",
                "mean_bid_depth",
                "mean_ask_volume",
                "mean_bid_volume",
            ]
        self.label_name = "ret"
        self.results = None

    def fit(self, y: pd.Series, X: pd.DataFrame):
        """ """

        if self.use_ob_feats:
            self.stored_y = y.values
            self.stored_X = X.values
            print(self.stored_X.shape)
            self.model = arch_model(
                y=self.stored_y,
                x=self.stored_X,
                vol=self.type,
                p=self.p,
                o=self.o,
                q=self.q,
                dist=self.dist,
                # rescale=True,
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
                # rescale=True,
            )

        self.results = self.model.fit(disp="off")
        print(self.results.summary())

    def forecast(self, steps: int = 1, X: pd.DataFrame = None):
        """
        Forecast the next steps volatility.
        """
        if self.results is None:
            raise ValueError("Model not fitted yet. Please fit the model first.")

        if self.use_ob_feats:
            X = X.values[-1, :].reshape(-1, 1)
            X = X[:, :, np.newaxis]
            print(X.shape)
            return self.results.forecast(horizon=steps, x=X).variance.values[0]  # [0]
        else:
            return self.results.forecast(horizon=steps).variance.values[0]  # [0]

    def update(self, new_y: pd.Series, new_X: pd.DataFrame):
        """
        Update the model with new data.
        """
        self.stored_y = np.append(self.stored_y, new_y.values)
        if self.use_ob_feats:
            self.stored_X = np.append(self.stored_X, new_X.values, axis=0)
            self.model = arch_model(
                y=self.stored_y,
                x=self.stored_X,
                vol=self.type,
                p=self.p,
                o=self.o,
                q=self.q,
                dist=self.dist,
                # rescale=True,
            )
        else:
            self.model = arch_model(
                y=self.stored_y,
                vol=self.type,
                p=self.p,
                o=self.o,
                q=self.q,
                dist=self.dist,
                # rescale=True,
            )

        self.results = self.model.fix(self.results.params.values)


class HARX(AbstractBenchmark):
    """
    HAR volatility predictor (we know already that vola is stationary).
    Given the past volatility, predict the next one using GARCH model.
    """

    def __init__(
        self,
        lags: list = [1, 6, 24],  # lags for HAR model
        l2: float = 70.0,  # shrinkage parameter, l2 regularization parameter
        is_weighted: bool = False,  # whether to weight points by the inverse of the vola
        use_ob_feats: bool = False,
        use_log_y: bool = True,
        *args,
        **kwargs,
    ):
        self.lags = lags
        self.l2 = l2
        self.is_weighted = is_weighted
        self.use_ob_feats = use_ob_feats
        self.use_log_y = use_log_y
        self.label_name = "log_vol" if use_log_y else "vol"
        self.name = "HARX" if use_ob_feats else "HAR"

        if self.use_log_y:
            self.ma_features = [f"log_vol_lag1_smooth{lag}" for lag in self.lags]
        else:
            self.ma_features = [f"vol_lag1_smooth{lag}" for lag in self.lags]

        if use_ob_feats:
            self.ob_features = [
                "log_volume",
                "mean_spread",
                "mean_bid_depth",
                "mean_ask_volume",
                "iq_range_bid_depth",
                "iq_range_ask_volume",
                "iq_range_ask_slope",
            ]
            self.feature_names = self.ma_features + self.ob_features
        else:
            self.ob_features = []
            self.feature_names = self.ma_features

        self.results = None

    def fit(self, y: pd.Series, X: pd.DataFrame):
        """ """

        X = X.loc[:, self.feature_names]

        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)

        X = sm.add_constant(X, has_constant="skip")

        if self.is_weighted:
            if self.use_log_y:
                weights = 1 / (0.0012 + np.exp(y.values))
            else:
                weights = 1 / (0.0012 + y.values)
        else:
            weights = np.ones(y.shape[0])

        # scale by sample size to make shrinkage uniform across different sample sizes
        normalized_weights = y.values.shape[0] * (weights / (weights.sum()))
        W = np.diag(normalized_weights)

        LambdaMat = self.l2 * np.diag(
            [0] * (1 + len(self.lags)) + [1] * (len(self.ob_features))
        )

        self.beta = np.linalg.inv(X.T @ (W @ X) + LambdaMat) @ (X.T @ (W @ y.values))
        self.results = self.beta

    def forecast(self, steps: int = 1, X: pd.DataFrame = None):
        """
        Forecast the next steps volatility.
        """
        if self.results is None:
            raise ValueError("Model not fitted yet. Please fit the model first.")

        X = X.loc[:, self.feature_names].iloc[-1:]  # take the last row for forecasting

        X = self.scaler.transform(X)

        # for a single observation we need `has_constant='add'`
        X = sm.add_constant(X, has_constant="add")

        if self.use_log_y:
            return np.array(np.exp(X @ self.beta))
        else:
            return np.array(X @ self.beta)

    def update(self, new_y: pd.Series, new_X: pd.DataFrame):
        """
        Update the model with new data.
        """
        pass  # not needed for this model


class ENET(AbstractBenchmark):
    """
    ElasticNet volatility predictor.
    """

    def __init__(
        self,
        alpha: float,
        l1_ratio: float,
        num_vola_lags: int = 10,
        use_ob_feats: bool = True,
        use_log_y: bool = True,
        *args,
        **kwargs,
    ):
        self.use_ob_feats = use_ob_feats
        self.use_log_y = use_log_y
        self.label_name = "log_vol" if use_log_y else "vol"
        self.name = "ENET"
        self.num_vola_lags = num_vola_lags
        if self.use_log_y:
            self.feature_names = [
                f"log_vol_lag{i}" for i in range(1, self.num_vola_lags)
            ]
        else:
            self.feature_names = [f"vol_lag{i}" for i in range(1, self.num_vola_lags)]
        if use_ob_feats:
            self.feature_names += [
                "log_volume",
                "mean_spread",
                "mean_bid_depth",
                "mean_ask_volume",
                "mean_bid_volume",
                "mean_weighted_spread",
                "mean_ask_slope",
                "mean_bid_slope",
                # 'trend_spread',
                # 'trend_bid_depth',
                # 'trend_ask_volume',
                # 'trend_bid_volume',
                # 'trend_weighted_spread',
                # 'trend_ask_slope',
                # 'trend_bid_slope',
                "iq_range_spread",
                "iq_range_bid_depth",
                "iq_range_ask_volume",
                "iq_range_bid_volume",
                "iq_range_weighted_spread",
                "iq_range_ask_slope",
                "iq_range_bid_slope",
            ]

        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.results = None

    def fit(self, y: pd.Series, X: pd.DataFrame):
        """ """
        X = X.loc[:, self.feature_names]

        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)

        self.model = ElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
        )
        self.model.fit(X, y)
        self.results = self.model

    def forecast(self, steps: int = 1, X: pd.DataFrame = None):
        """
        Forecast the next steps volatility.
        """
        if self.results is None:
            raise ValueError("Model not fitted yet. Please fit the model first.")
        # X = X.loc[:, self.feature_names].values[-1, :].reshape(1, -1)#.iloc[-1:]  # take the last row for forecasting

        X = X.loc[:, self.feature_names].iloc[-1:]  # take the last row for forecasting
        X = self.scaler.transform(X)

        # X = X.values[-1, :].reshape(1, -1)
        # X = self.scalerX.transform(X)

        pred = self.model.predict(X)  # [0]
        return np.exp(pred)

    def update(self, new_y: pd.Series, new_X: pd.DataFrame):
        """
        Update the model with new data.
        """
        # self.stored_y = np.append(self.stored_y, np.log(new_y.values))
        # self.stored_X = np.append(self.stored_X, self.scalerX.transform(new_X.values[-1, :].reshape(1, -1)), axis=0)
        pass


class RF(AbstractBenchmark):
    """ """

    def __init__(
        self,
        n_estimators=100,
        max_depth=10,
        min_samples_split=2,
        num_vola_lags: int = 10,
        use_ob_feats: bool = True,
        use_log_y: bool = True,
        *args,
        **kwargs,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

        self.use_ob_feats = use_ob_feats
        self.use_log_y = use_log_y
        self.label_name = "log_vol" if use_log_y else "vol"
        self.name = "RF"
        self.num_vola_lags = num_vola_lags
        if self.use_log_y:
            self.feature_names = [
                f"log_vol_lag{i}" for i in range(1, self.num_vola_lags)
            ]
        else:
            self.feature_names = [f"vol_lag{i}" for i in range(1, self.num_vola_lags)]
        if use_ob_feats:
            self.feature_names += [
                "log_volume",
                "mean_spread",
                "mean_bid_depth",
                "mean_ask_volume",
                "mean_bid_volume",
                "mean_weighted_spread",
                "mean_ask_slope",
                "mean_bid_slope",
                "iq_range_spread",
                "iq_range_bid_depth",
                "iq_range_ask_volume",
                "iq_range_bid_volume",
                "iq_range_weighted_spread",
                "iq_range_ask_slope",
                "iq_range_bid_slope",
            ]

        self.results = None

    def fit(self, y: pd.Series, X: pd.DataFrame):
        """ """
        X = X.loc[:, self.feature_names]

        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)

        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
        )
        self.model.fit(X, y)
        self.results = self.model

    def forecast(self, steps: int = 1, X: pd.DataFrame = None):
        """
        Forecast the next steps volatility.
        """
        if self.results is None:
            raise ValueError("Model not fitted yet. Please fit the model first.")
        # X = X.loc[:, self.feature_names].values[-1, :].reshape(1, -1)#.iloc[-1:]  # take the last row for forecasting

        X = X.loc[:, self.feature_names].iloc[-1:]  # take the last row for forecasting
        X = self.scaler.transform(X)

        # X = X.values[-1, :].reshape(1, -1)
        # X = self.scalerX.transform(X)

        pred = self.model.predict(X)  # [0]
        return np.exp(pred)

    def update(self, new_y: pd.Series, new_X: pd.DataFrame):
        """
        Update the model with new data.
        """
        # self.stored_y = np.append(self.stored_y, np.log(new_y.values))
        # self.stored_X = np.append(self.stored_X, self.scalerX.transform(new_X.values[-1, :].reshape(1, -1)), axis=0)
        pass


class XGBM(AbstractBenchmark):
    """ """

    def __init__(
        self,
        n_estimators=100,
        max_depth=10,
        min_samples_split=2,
        num_vola_lags: int = 10,
        use_ob_feats: bool = True,
        use_log_y: bool = True,
        *args,
        **kwargs,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

        self.use_ob_feats = use_ob_feats
        self.use_log_y = use_log_y
        self.label_name = "log_vol" if use_log_y else "vol"
        self.name = "XGBM"
        self.num_vola_lags = num_vola_lags
        if self.use_log_y:
            self.feature_names = [
                f"log_vol_lag{i}" for i in range(1, self.num_vola_lags)
            ]
        else:
            self.feature_names = [f"vol_lag{i}" for i in range(1, self.num_vola_lags)]
        if use_ob_feats:
            self.feature_names += [
                "log_volume",
                "mean_spread",
                "mean_bid_depth",
                "mean_ask_volume",
                "mean_bid_volume",
                "mean_weighted_spread",
                "mean_ask_slope",
                "mean_bid_slope",
                "iq_range_spread",
                "iq_range_bid_depth",
                "iq_range_ask_volume",
                "iq_range_bid_volume",
                "iq_range_weighted_spread",
                "iq_range_ask_slope",
                "iq_range_bid_slope",
            ]

        self.results = None

    def fit(self, y: pd.Series, X: pd.DataFrame):
        """ """
        X = X.loc[:, self.feature_names]

        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)

        self.model = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
        )
        self.model.fit(X, y)
        self.results = self.model

    def forecast(self, steps: int = 1, X: pd.DataFrame = None):
        """
        Forecast the next steps volatility.
        """
        if self.results is None:
            raise ValueError("Model not fitted yet. Please fit the model first.")
        # X = X.loc[:, self.feature_names].values[-1, :].reshape(1, -1)#.iloc[-1:]  # take the last row for forecasting

        X = X.loc[:, self.feature_names].iloc[-1:]  # take the last row for forecasting
        X = self.scaler.transform(X)

        # X = X.values[-1, :].reshape(1, -1)
        # X = self.scalerX.transform(X)

        pred = self.model.predict(X)  # [0]
        return np.exp(pred)

    def update(self, new_y: pd.Series, new_X: pd.DataFrame):
        """
        Update the model with new data.
        """
        # self.stored_y = np.append(self.stored_y, np.log(new_y.values))
        # self.stored_X = np.append(self.stored_X, self.scalerX.transform(new_X.values[-1, :].reshape(1, -1)), axis=0)
        pass
