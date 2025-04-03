from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.tools import diff
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np

#Plots acf and pacf side by side
def plot_acf_pacf(df):
    fig, ax = plt.subplots(1,2)
    sm.graphics.tsa.plot_acf(df, ax=ax[0])
    sm.graphics.tsa.plot_pacf(df, ax=ax[1])
    plt.show()


#Determines the optimal d using the Augmented Dickey–Fuller test 
def determine_d(df, alpha, max_d):
    d = 0
    current_df = df.copy()
    while d <= max_d:
        adf_test = adfuller(current_df)
        if adf_test[1] < alpha:
            return d
        current_df = current_df.diff().dropna()
        d += 1
    return d


#Determines the p and q that minimizes the AIC
def find_p_q(ts, d, max_p, max_q, exog, exog_df):
    p_q_table = pd.DataFrame(index=range(max_p + 1), columns=range(max_q + 1))
    if exog == True:
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                model = ARIMA(ts, exog=exog_df, order=(p, d, q))
                fit = model.fit(method_kwargs={'maxiter':1000})
                p_q_table.loc[p, q] = fit.aic
        return p_q_table
    else:
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                model = ARIMA(ts, order=(p, d, q))
                fit = model.fit(method_kwargs={'maxiter':1000})
                p_q_table.loc[p, q] = fit.aic
        return p_q_table

#Finds and fits the most suitible ARIMA/ARIMAX model
def find_fit(ts, alpha, max_d, max_p, max_q, exog, exog_df = None):
    d = determine_d(ts, alpha, max_d)
    table = find_p_q(ts, d, max_p, max_q, exog, exog_df)
    best_p, best_q = divmod(table.values.argmin(), table.shape[1])
    if exog == True:
        best_model = ARIMA(ts, exog_df, order=(best_p, d, best_q))
    else:
        best_model = ARIMA(train, order=(best_p, d, best_q))
    return best_model.fit(method_kwargs={'maxiter':1000})

#Plots the residuals including their density
def plot_residuals(model):
    resids = model.resid[1:]
    fig, ax = plt.subplots(1,2)
    resids.plot(title='Residuals', ax=ax[0])
    resids.plot(title='Density', kind = 'kde', ax = ax[1])
    plt.show()


'''
#Test data set I used when implementing the model

from pmdarima.datasets.stocks import load_msft

data_set = load_msft()

df_open = data_set["Open"]
log_open = np.log(df_open)
split_index = int(len(df_open) * 0.7)
train = log_open[:split_index]
test = log_open[split_index:]

best_fit = find_fit(train, 0.01, 5, 5, 5, False)

train_pred = best_fit.predict(start = 1, end = len(train) - 1)
forecast_returns = best_fit.forecast(steps = len(test))

# Plot
plt.plot(train.index, train, label="Training Data", color="blue")
plt.plot(test.index, test, label="True Test Values", color="green")
plt.plot(forecast_returns.index, forecast_returns, label="Out-of-Sample Forecast", color="purple", linestyle="-")
plt.title("ARIMA Model: True vs. Predicted Values (Log Returns)")
plt.legend()
plt.show()
'''
