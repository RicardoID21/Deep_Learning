# EL sharpe es ANUAL, tenemos que anualizarlo porque está de 5 minutos

from ta.trend import EMAIndicator, SMAIndicator
from ta.volatility import BollingerBands
import numpy as np

# --- Métricas ---
def calculate_returns(portafolio_value):
    return np.diff(portafolio_value) / portafolio_value[:-1]

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    if returns.std() == 0:
        return 0
    excess_returns = returns - risk_free_rate / 252
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

def calculate_sortino_ratio(returns, risk_free_rate=0.0):
    downside_returns = returns[returns < 0]
    if downside_returns.std() == 0:
        return 0
    excess_returns = returns - risk_free_rate / 252
    return np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252)

def calculate_calmar_ratio(portafolio_value):
    returns = calculate_returns(portafolio_value)
    peak = np.maximum.accumulate(portafolio_value)
    drawdown = (portafolio_value - peak) / peak
    max_drawdown = np.min(drawdown)
    annual_return = np.mean(returns) * 252
    return annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

def calculate_win_loss_ratio(transacciones, data):
    wins = 0
    losses = 0
    for tx in transacciones:
        entry_price = tx['Precio']
        exit_index = tx.get('Exit')
        if exit_index is None or exit_index not in data.index:
            continue
        exit_price = data.loc[exit_index, 'Close']
        pnl = (exit_price - entry_price) if tx['Tipo'] else (entry_price - exit_price)
        if pnl > 0:
            wins += 1
        elif pnl < 0:
            losses += 1
    total = wins + losses
    return wins / total if total > 0 else 0

# --- Función principal ---
def objective_func(trial, data):
    ema_optimo = trial.suggest_int('ema_window', 9, 50)
    bb_window = trial.suggest_int('bb_window', 10, 40)
    bb_dev = trial.suggest_float('bb_dev', 1.5, 3.5)
    sma_optimo = trial.suggest_int('sma_window', 9, 30)

    stop_loss = trial.suggest_float('stop_loss', 0.01, 0.2)
    take_profit = trial.suggest_float('take_profit', 0.01, 0.2)
    n_shares = trial.suggest_categorical('n_shares', [4000, 4500, 5000, 5500, 6000, 6500, 7000])

    # Indicadores técnicos
    data['EMA_20'] = EMAIndicator(close=data['Close'], window=ema_optimo).ema_indicator()
    bb = BollingerBands(close=data['Close'], window=bb_window, window_dev=bb_dev)
    data['BB_upper'] = bb.bollinger_hband()
    data['BB_middle'] = bb.bollinger_mavg()
    data['BB_lower'] = bb.bollinger_lband()
    data['SMA_20'] = SMAIndicator(close=data['Close'], window=sma_optimo).sma_indicator()
    data = data.dropna()

    # Generar señales
    data['EMA_Signal'] = data.apply(
        lambda row: 1 if row['Close'] > row['EMA_20'] and data['Close'].shift(1)[row.name] < data['EMA_20'].shift(1)[row.name]
        else (-1 if row['Close'] < row['EMA_20'] and data['Close'].shift(1)[row.name] > data['EMA_20'].shift(1)[row.name] else 0),
        axis=1
    )
    data['BB_Signal'] = 0
    data.loc[data['Close'] < data['BB_lower'], 'BB_Signal'] = 1
    data.loc[data['Close'] > data['BB_upper'], 'BB_Signal'] = -1


    data['SMA_Signal'] = data.apply(
        lambda row: 1 if row['Close'] > row['SMA_20'] and data['Close'].shift(1)[row.name] < data['SMA_20'].shift(1)[row.name]
        else (-1 if row['Close'] < row['SMA_20'] and data['Close'].shift(1)[row.name] > data['SMA_20'].shift(1)[row.name] else 0),
        axis=1
    )

    # Señal combinada
    signal_sum = data[['EMA_Signal', 'BB_Signal', 'SMA_Signal']].sum(axis=1)
    data['Cash_medio'] = 0
    data.loc[signal_sum == 2, 'Cash_medio'] = 1
    data.loc[signal_sum == -2, 'Cash_medio'] = -1

    # Simulación
    capital = 1_000_000
    comision = 0.00125
    transacciones = []
    portafolio_value = []
    active_position = {}

    for i in data.index:
        precio_actual = data.loc[i, 'Close']

        if not active_position:
            if data.loc[i, 'Cash_medio'] == 1:
                active_position = {'Date': i, 'Precio': precio_actual, 'Tipo': True}
            elif data.loc[i, 'Cash_medio'] == -1:
                active_position = {'Date': i, 'Precio': precio_actual, 'Tipo': False}

        if active_position:
            precio_entrada = active_position['Precio']
            tipo = active_position['Tipo']

            if tipo:
                target_value = precio_entrada * (1 + take_profit)
                max_loss = precio_entrada * (1 - stop_loss)
                if precio_actual >= target_value or precio_actual <= max_loss:
                    capital += (precio_actual * n_shares) * (1 - comision)
                    active_position['Exit'] = i
                    transacciones.append(active_position)
                    active_position = {}
            else:
                target_value = precio_entrada * (1 - take_profit)
                max_loss = precio_entrada * (1 + stop_loss)
                if precio_actual <= target_value or precio_actual >= max_loss:
                    capital += ((precio_entrada - precio_actual) * n_shares) * (1 - comision)
                    active_position['Exit'] = i
                    transacciones.append(active_position)
                    active_position = {}

        # Valor del portafolio
        if not active_position:
            portafolio_value.append(capital)
        else:
            if active_position['Tipo']:
                valor_posicion = (precio_actual - active_position['Precio']) * n_shares
            else:
                valor_posicion = (active_position['Precio'] - precio_actual) * n_shares
            portafolio_value.append(capital + valor_posicion)

    if len(portafolio_value) < 2:
        return 0

    returns = calculate_returns(np.array(portafolio_value))
    sharpe = calculate_sharpe_ratio(returns)
    sortino = calculate_sortino_ratio(returns)
    calmar = calculate_calmar_ratio(np.array(portafolio_value))
    win_loss = calculate_win_loss_ratio(transacciones, data)
    final_value = portafolio_value[-1]

    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "win_loss": win_loss,
        "final_portfolio_value": final_value
    }
