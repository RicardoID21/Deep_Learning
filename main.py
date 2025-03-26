from functions import backtesting
import pandas as pd
import numpy as np
from ta.trend import EMAIndicator
from ta.volatility import BollingerBands
from ta.trend import SMAIndicator

# Carga de Datos
url_train = "aapl_5m_train.csv"
train = pd.read_csv(url_train)

url_test = "aapl_5m_test.csv"
test = pd.read_csv(url_test)

train = train[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]
train.set_index(train.columns[0], inplace=True)

# EMA
ema = EMAIndicator(close=train['Close'], window=20)
train['EMA_20'] = ema.ema_indicator()

# Bollinger Bands
bb = BollingerBands(close=train['Close'], window=20, window_dev=2)
train['BB_upper'] = bb.bollinger_hband()
train['BB_middle'] = bb.bollinger_mavg()
train['BB_lower'] = bb.bollinger_lband()

# SMA
sma = SMAIndicator(close=train['Close'], window=20)
train['SMA_20'] = sma.sma_indicator()

train = train.dropna()

train['EMA_Signal'] = 0
train['EMA_Signal'] = train.apply(
    lambda row: 1 if row['Close'] > row['EMA_20'] and train['Close'].shift(1)[row.name] < train['EMA_20'].shift(1)[row.name]
    else (-1 if row['Close'] < row['EMA_20'] and train['Close'].shift(1)[row.name] > train['EMA_20'].shift(1)[row.name]
          else 0), axis=1
)

train['BB_Signal'] = 0
train.loc[train['Close'] < train['BB_lower'], 'BB_Signal'] = 1
train.loc[train['Close'] > train['BB_upper'], 'BB_Signal'] = -1


train['SMA_Signal'] = 0
train['SMA_Signal'] = train.apply(
    lambda row: 1 if row['Close'] > row['SMA_20'] and train['Close'].shift(1)[row.name] < train['SMA_20'].shift(1)[row.name]
    else (-1 if row['Close'] < row['SMA_20'] and train['Close'].shift(1)[row.name] > train['SMA_20'].shift(1)[row.name]
          else 0), axis=1
)

train['Signal_Cash'] = 0
for i in train.index:
    if train.loc[i, 'EMA_Signal'] == 1 and train.loc[i, 'BB_Signal'] == 1 and train.loc[i, 'SMA_Signal'] == 1:
        train.loc[i, 'Signal_Cash'] = 1
    elif train.loc[i, 'EMA_Signal'] == -1 and train.loc[i, 'BB_Signal'] == -1 and train.loc[i, 'SMA_Signal'] == -1:
        train.loc[i, 'Signal_Cash'] = -1

train['Cash_medio'] = 0
signal_sum = train[['EMA_Signal', 'BB_Signal', 'SMA_Signal']].sum(axis=1)


train.loc[signal_sum == 2, 'Cash_medio'] = 1
train.loc[signal_sum == -2, 'Cash_medio'] = -1


print(train['Cash_medio'].value_counts())





capital = 1_000_000
t_profit = 0.10
t_losses = 0.20
n_shares = 2_500

#portafolio_value = backtesting(capital, t_profit, t_losses, n_shares, train)