from ta.trend import EMAIndicator, SMAIndicator
from ta.volatility import BollingerBands
import numpy as np

# --- Metrics ---

def calculate_returns(portfolio_value):
    return np.diff(portfolio_value) / portfolio_value[:-1]

def calculate_sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=19656):
    if returns.std() == 0:
        return 0
    excess_returns = returns - risk_free_rate / periods_per_year
    sharpe = np.mean(excess_returns) / np.std(excess_returns)
    return sharpe * np.sqrt(periods_per_year)

def calculate_sortino_ratio(returns, risk_free_rate=0.0, periods_per_year=19656):
    downside_returns = returns[returns < 0]
    if downside_returns.std() == 0:
        return 0
    excess_returns = returns - risk_free_rate / periods_per_year
    sortino = np.mean(excess_returns) / np.std(downside_returns)
    return sortino * np.sqrt(periods_per_year)

def calculate_calmar_ratio(portfolio_value, periods_per_year=19656):
    returns = calculate_returns(portfolio_value)
    peak = np.maximum.accumulate(portfolio_value)
    drawdown = (portfolio_value - peak) / peak
    max_drawdown = np.min(drawdown)
    annual_return = np.mean(returns) * periods_per_year
    return annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

def calculate_win_loss_ratio(trades, data):
    wins = 0
    losses = 0
    for trade in trades:
        entry_price = trade['Price']
        exit_index = trade.get('Exit')
        if exit_index is None or exit_index not in data.index:
            continue
        exit_price = data.loc[exit_index, 'Close']
        pnl = (exit_price - entry_price) if trade['Type'] else (entry_price - exit_price)
        if pnl > 0:
            wins += 1
        elif pnl < 0:
            losses += 1
    total = wins + losses
    return wins / total if total > 0 else 0

# --- Objective Function for Optuna ---

def objective_func(trial, data):
    from ta.trend import EMAIndicator, SMAIndicator
    from ta.volatility import BollingerBands

    import numpy as np

    data = data.copy()

    # Suggest hyperparameters
    ema_window = trial.suggest_int('ema_window', 9, 50)
    bb_window = trial.suggest_int('bb_window', 10, 40)
    bb_dev = trial.suggest_float('bb_dev', 1.5, 3.5)
    sma_window = trial.suggest_int('sma_window', 9, 30)
    stop_loss = trial.suggest_float('stop_loss', 0.01, 0.2)
    take_profit = trial.suggest_float('take_profit', 0.01, 0.2)
    n_shares = trial.suggest_categorical('n_shares', [4000, 4500, 5000, 5500, 6000, 6500, 7000])

    # Indicators
    data['EMA_20'] = EMAIndicator(close=data['Close'], window=ema_window).ema_indicator()
    bb = BollingerBands(close=data['Close'], window=bb_window, window_dev=bb_dev)
    data['BB_upper'] = bb.bollinger_hband()
    data['BB_middle'] = bb.bollinger_mavg()
    data['BB_lower'] = bb.bollinger_lband()
    data['SMA_20'] = SMAIndicator(close=data['Close'], window=sma_window).sma_indicator()
    data = data.dropna()

    # Signals
    data['EMA_Signal'] = 0
    data.loc[(data['Close'] > data['EMA_20']) & (data['Close'].shift(1) <= data['EMA_20'].shift(1)), 'EMA_Signal'] = 1
    data.loc[(data['Close'] < data['EMA_20']) & (data['Close'].shift(1) >= data['EMA_20'].shift(1)), 'EMA_Signal'] = -1

    data['BB_Signal'] = 0
    data.loc[data['Close'] < data['BB_lower'], 'BB_Signal'] = 1
    data.loc[data['Close'] > data['BB_upper'], 'BB_Signal'] = -1

    data['SMA_Signal'] = 0
    long_condition = (data['Close'] > data['SMA_20']) & (data['Close'].shift(1) <= data['SMA_20'].shift(1))
    short_condition = (data['Close'] < data['SMA_20']) & (data['Close'].shift(1) >= data['SMA_20'].shift(1))
    data.loc[long_condition, 'SMA_Signal'] = 1
    data.loc[short_condition, 'SMA_Signal'] = -1

    # Combine signals
    signal_sum = data[['EMA_Signal', 'BB_Signal', 'SMA_Signal']].sum(axis=1)
    data['Signal'] = 0
    data.loc[signal_sum == 2, 'Signal'] = 1
    data.loc[signal_sum == -2, 'Signal'] = -1

    # Backtest
    capital = 1_000_000
    commission = 0.00125
    trades = []
    portfolio_value = []
    active_position = {}

    for i in data.index:
        current_price = data.loc[i, 'Close']

        # Entry
        if not active_position:
            if data.loc[i, 'Signal'] == 1:
                active_position = {
                    'Date': i,
                    'Price': current_price,
                    'Type': True,  # Long
                    'Shares': n_shares,
                    'Investment': current_price * n_shares * (1 + commission)
                }
                capital -= active_position['Investment']
            elif data.loc[i, 'Signal'] == -1:
                active_position = {
                    'Date': i,
                    'Price': current_price,
                    'Type': False,  # Short
                    'Shares': n_shares,
                    'Investment': current_price * n_shares * (1 + commission)
                }
                capital -= active_position['Investment']

        # Exit
        if active_position:
            entry_price = active_position['Price']
            is_long = active_position['Type']
            shares = active_position['Shares']

            if is_long:
                target = entry_price * (1 + take_profit)
                stop = entry_price * (1 - stop_loss)
                if current_price >= target or current_price <= stop:
                    exit_value = current_price * shares * (1 - commission)
                    capital += exit_value
                    pnl = exit_value - active_position['Investment']
                    active_position.update({'Exit': i, 'PnL': pnl})
                    trades.append(active_position)
                    active_position = {}
            else:
                target = entry_price * (1 - take_profit)
                stop = entry_price * (1 + stop_loss)
                if current_price <= target or current_price >= stop:
                    exit_value = (entry_price - current_price) * shares * (1 - commission)
                    capital += active_position['Investment'] + exit_value
                    pnl = exit_value
                    active_position.update({'Exit': i, 'PnL': pnl})
                    trades.append(active_position)
                    active_position = {}

        # Portfolio valuation
        if active_position:
            if active_position['Type']:
                unrealized = (current_price - active_position['Price']) * n_shares
            else:
                unrealized = (active_position['Price'] - current_price) * n_shares
            portfolio_value.append(capital + active_position['Investment'] + unrealized)
        else:
            portfolio_value.append(capital)

    # Metrics
    if len(portfolio_value) < 2:
        return {
            "sharpe": 0,
            "sortino": 0,
            "calmar": 0,
            "win_loss": 0,
            "final_portfolio_value": capital
        }

    returns = calculate_returns(np.array(portfolio_value))
    sharpe = calculate_sharpe_ratio(returns)
    sortino = calculate_sortino_ratio(returns)
    calmar = calculate_calmar_ratio(np.array(portfolio_value))
    win_loss = calculate_win_loss_ratio(trades, data)
    final_value = portfolio_value[-1]

    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "win_loss": win_loss,
        "final_portfolio_value": final_value
    }

def evaluate_with_params(params, data):
    # Clone the data to avoid modifying the original DataFrame
    data = data.copy()

    # Extract parameters
    ema_window = params['ema_window']
    bb_window = params['bb_window']
    bb_dev = params['bb_dev']
    sma_window = params['sma_window']
    stop_loss = params['stop_loss']
    take_profit = params['take_profit']
    n_shares = params['n_shares']

    # Calculate technical indicators
    data['EMA_20'] = EMAIndicator(close=data['Close'], window=ema_window).ema_indicator()
    bb = BollingerBands(close=data['Close'], window=bb_window, window_dev=bb_dev)
    data['BB_upper'] = bb.bollinger_hband()
    data['BB_middle'] = bb.bollinger_mavg()
    data['BB_lower'] = bb.bollinger_lband()
    data['SMA_20'] = SMAIndicator(close=data['Close'], window=sma_window).sma_indicator()
    data = data.dropna()

    # Generate indicator signals
    data['EMA_Signal'] = 0
    data.loc[(data['Close'] > data['EMA_20']) & (data['Close'].shift(1) <= data['EMA_20'].shift(1)), 'EMA_Signal'] = 1
    data.loc[(data['Close'] < data['EMA_20']) & (data['Close'].shift(1) >= data['EMA_20'].shift(1)), 'EMA_Signal'] = -1

    data['BB_Signal'] = 0
    data.loc[data['Close'] < data['BB_lower'], 'BB_Signal'] = 1
    data.loc[data['Close'] > data['BB_upper'], 'BB_Signal'] = -1

    data['SMA_Signal'] = 0
    long_condition = (data['Close'] > data['SMA_20']) & (data['Close'].shift(1) <= data['SMA_20'].shift(1))
    short_condition = (data['Close'] < data['SMA_20']) & (data['Close'].shift(1) >= data['SMA_20'].shift(1))
    data.loc[long_condition, 'SMA_Signal'] = 1
    data.loc[short_condition, 'SMA_Signal'] = -1

    # Combine all signals into a single trading signal
    signal_sum = data[['EMA_Signal', 'BB_Signal', 'SMA_Signal']].sum(axis=1)
    data['Signal'] = 0
    data.loc[signal_sum == 2, 'Signal'] = 1
    data.loc[signal_sum == -2, 'Signal'] = -1

    # Backtest simulation
    capital = 1_000_000
    commission = 0.00125
    trades = []
    portfolio_value = []
    active_position = {}

    for i in data.index:
        current_price = data.loc[i, 'Close']

        # Open position if there's a signal and no active position
        if not active_position:
            if data.loc[i, 'Signal'] == 1:
                active_position = {
                    'Date': i,
                    'Price': current_price,
                    'Type': True,  # Long
                    'Shares': n_shares,
                    'Investment': current_price * n_shares * (1 + commission)
                }
                capital -= active_position['Investment']
            elif data.loc[i, 'Signal'] == -1:
                active_position = {
                    'Date': i,
                    'Price': current_price,
                    'Type': False,  # Short
                    'Shares': n_shares,
                    'Investment': current_price * n_shares * (1 + commission)
                }
                capital -= active_position['Investment']

        # Manage exit if active position exists
        if active_position:
            entry_price = active_position['Price']
            is_long = active_position['Type']
            shares = active_position['Shares']

            if is_long:
                target = entry_price * (1 + take_profit)
                stop = entry_price * (1 - stop_loss)
                if current_price >= target or current_price <= stop:
                    exit_value = current_price * shares * (1 - commission)
                    capital += exit_value
                    pnl = exit_value - active_position['Investment']
                    active_position.update({'Exit': i, 'PnL': pnl})
                    trades.append(active_position)
                    active_position = {}
            else:
                target = entry_price * (1 - take_profit)
                stop = entry_price * (1 + stop_loss)
                if current_price <= target or current_price >= stop:
                    exit_value = (entry_price - current_price) * shares * (1 - commission)
                    capital += active_position['Investment'] + exit_value
                    pnl = exit_value
                    active_position.update({'Exit': i, 'PnL': pnl})
                    trades.append(active_position)
                    active_position = {}

        # Portfolio valuation
        if active_position:
            if active_position['Type']:
                unrealized = (current_price - active_position['Price']) * n_shares
            else:
                unrealized = (active_position['Price'] - current_price) * n_shares
            portfolio_value.append(capital + active_position['Investment'] + unrealized)
        else:
            portfolio_value.append(capital)

    # Final fallback if not enough data
    if len(portfolio_value) < 2:
        return {
            "return": portfolio_value,
            "sharpe": 0,
            "sortino": 0,
            "calmar": 0,
            "win_loss": 0,
            "final_portfolio_value": capital
        }

    # Final metric calculation
    returns = calculate_returns(np.array(portfolio_value))
    sharpe = calculate_sharpe_ratio(returns)
    sortino = calculate_sortino_ratio(returns)
    calmar = calculate_calmar_ratio(np.array(portfolio_value))
    win_loss = calculate_win_loss_ratio(trades, data)
    final_value = portfolio_value[-1]

    return {
        "return": portfolio_value,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "win_loss": win_loss,
        "final_portfolio_value": final_value
    }