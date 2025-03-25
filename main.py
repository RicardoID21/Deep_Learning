from functions import backtesting
import pandas as pd

# Carga de Datos
url_train = "https://raw.githubusercontent.com/RicardoID21/Deep_Learning/main/aapl_5m_train.csv"
train = pd.read_csv(url_train)

url_test = "https://raw.githubusercontent.com/RicardoID21/Deep_Learning/main/aapl_5m_test.csv"
test = pd.read_csv(url_test)

capital = 1_000_000
t_profit = 0.10
t_losses = 0.20
n_shares = 2_500

portafolio_value = backtesting(capital, t_profit, t_losses, n_shares, train)