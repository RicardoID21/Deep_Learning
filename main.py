from functions import backtesting
import pandas as pd
import numpy as np

# Carga de Datos
url_train = "aapl_5m_train.csv"
train = pd.read_csv(url_train)

url_test = "aapl_5m_test.csv"
test = pd.read_csv(url_test)

train = train[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]
train.set_index(train.columns[0], inplace=True)

capital = 1_000_000
t_profit = 0.10
t_losses = 0.20
n_shares = 2_500

#portafolio_value = backtesting(capital, t_profit, t_losses, n_shares, train)