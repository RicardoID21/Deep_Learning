import numpy as np
import pandas as pd
from ta.trend import EMAIndicator
from ta.volatility import BollingerBands
from ta.trend import SMAIndicator

from main import train


#RSI, MEDIA MOVIL, ICHIMOKUO

# EL sharpe es ANUAL, tenemos que anualizarlo porqeu está de 5 minutos

def backtesting(capital, t_profit, t_losses, n_shares, data, params):
    # Definir Parametros
    rsi, fibonachi, ichimokuo = params

    # Historial de Transacciones
    transacciones = []

    for i in data:
        if len(active_position) == 0:
            #Validar SEÑAL AQUI MUEVELE RICHIE, HUGO, añadan cuantos if necesiten de forma que entren las señales de trading del indicador o si desean combinar varias
            if fibonachi >= 80: # Para un LONG
                active_position = {'Date': data.index, 'Precio': data[i], 'Tipo': True}

            if fibonachi <= 10: # Para un SHORT
                active_position = {'Date': data.index, 'Precio': data[i], 'Tipo': False}

            if ichimokuo >= 80: # Para un LONG
                active_position = {'Date': data.index, 'Precio': data[i], 'Tipo': True}

        # Revisar si hay posicion abierta y checar el Tipo de Posicion
        if len(active_position) > 0:
            if active_position.Tipo == True:
                l = active_position.Precio
                target_value = l * (1 + t_profit)
                max_loss = l * (1 - t_losses)
            elif active_position.Tipo == False:
                s = active_position.Precio
                # Para posiciones cortas
                target_value_s = s * (1 - t_profit)
                max_loss_s = s * (1 + max_loss)

            today = data[i]

            if today >= target_value or today <= max_loss:
                long_position = today * n_shares
                capital += long_position
                transacciones.append(active_position)
                active_position = []

            if today <= target_value_s or today >= max_loss_s:
                short_position = (s - today) * n_shares
                capital += short_position
                transacciones.append(active_position)
                active_position = []

        else:
            continue



def objective_func(trial, data):

    ema_optimo = trial.suggest_int('ema_window', 9,50)
    bb_window = trial.suggest_int('bb_window',10 ,40)
    bb_dev = trial.suggest_float('bb_deb', 1.5, 3.5)
    sma_optimo = trial.suggest_int('sma_window', 9, 30)

    stop_loss = trial.suggest_float('stop_loss', 0.01, 0.2)
    take_profit= trial.suggest_float('take_profit', 0.01, 0.2)
    n_shares = trial.suggest_categorical('n_shares', [4000, 4500, 5000, 5500, 6000, 6500, 7000])


    # EMA
    ema = EMAIndicator(close=data['Close'], window=ema_optimo)
    data['EMA_20'] = ema.ema_indicator()

    # Bollinger Bands
    bb = BollingerBands(close=data['Close'], window=bb_window, window_dev=bb_dev)
    data['BB_upper'] = bb.bollinger_hband()
    data['BB_middle'] = bb.bollinger_mavg()
    data['BB_lower'] = bb.bollinger_lband()

    # SMA
    sma = SMAIndicator(close=data['Close'], window=sma_optimo)
    data['SMA_20'] = sma.sma_indicator()

    data = data.dropna()

    data['EMA_Signal'] = 0
    data['EMA_Signal'] = data.apply(
        lambda row: 1 if row['Close'] > row['EMA_20'] and data['Close'].shift(1)[row.name] < data['EMA_20'].shift(1)[
            row.name]
        else (-1 if row['Close'] < row['EMA_20'] and data['Close'].shift(1)[row.name] > data['EMA_20'].shift(1)[
            row.name]
              else 0), axis=1
    )

    data['BB_Signal'] = 0
    data.loc[data['Close'] < data['BB_lower'], 'BB_Signal'] = 1
    data.loc[data['Close'] > data['BB_upper'], 'BB_Signal'] = -1

    data['SMA_Signal'] = 0
    data['SMA_Signal'] = data.apply(
        lambda row: 1 if row['Close'] > row['SMA_20'] and data['Close'].shift(1)[row.name] < data['SMA_20'].shift(1)[
            row.name]
        else (-1 if row['Close'] < row['SMA_20'] and data['Close'].shift(1)[row.name] > data['SMA_20'].shift(1)[
            row.name]
              else 0), axis=1
    )

    data['Signal_Cash'] = 0
    for i in data.index:
        if data.loc[i, 'EMA_Signal'] == 1 and data.loc[i, 'BB_Signal'] == 1 and data.loc[i, 'SMA_Signal'] == 1:
            data.loc[i, 'Signal_Cash'] = 1
        elif data.loc[i, 'EMA_Signal'] == -1 and data.loc[i, 'BB_Signal'] == -1 and data.loc[i, 'SMA_Signal'] == -1:
            data.loc[i, 'Signal_Cash'] = -1

    data['Cash_medio'] = 0
    signal_sum = data[['EMA_Signal', 'BB_Signal', 'SMA_Signal']].sum(axis=1)

    data.loc[signal_sum == 2, 'Cash_medio'] = 1
    data.loc[signal_sum == -2, 'Cash_medio'] = -1
