import numpy as np
import optuna
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from functions import objective_func, evaluate_with_params

# Load Data
url_train = "aapl_5m_train.csv"
train = pd.read_csv(url_train)

url_test = "aapl_5m_test.csv"
test = pd.read_csv(url_test)

# Clean and set datetime index
train = train[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]
train.set_index(train.columns[0], inplace=True)

test = test[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]
test.set_index(test.columns[0], inplace=True)

# Run the Optuna study (optimize for Sharpe ratio)
results = []

def wrapped_objective(trial):
    result = objective_func(trial, train)
    results.append(result)
    return result["sharpe"] if not np.isnan(result["sharpe"]) else -np.inf

study = optuna.create_study(direction="maximize")
study.optimize(wrapped_objective, n_trials=50)

# Print best result
print("\n🔍 Best Sharpe Ratio:", round(study.best_value, 4))
print("⚙️  Best Parameters:", study.best_params)

# Extract metrics of best result
best_index = max(
    range(len(results)),
    key=lambda i: results[i]["sharpe"] if not np.isnan(results[i]["sharpe"]) else -np.inf
)
best_result = results[best_index]

print("\n📊 Metrics of the Best Result:")
for k, v in best_result.items():
    print(f"{k:<25}: {round(v, 4) if isinstance(v, float) else v}")

# Evaluate performance on train and test datasets
train_result = evaluate_with_params(study.best_params, train)
test_result = evaluate_with_params(study.best_params, test)

# Align portfolio values with time index
test_portfolio = pd.Series(test_result['return'], index=test.iloc[-len(test_result['return']):].index)
train_portfolio = pd.Series(train_result['return'], index=train.iloc[-len(train_result['return']):].index)

# Convert index to datetime
test_portfolio.index = pd.to_datetime(test_portfolio.index)
train_portfolio.index = pd.to_datetime(train_portfolio.index)

# Plot test performance
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(test_portfolio, label="Portfolio Value (Test)", color='black')
ax.set_title("Portfolio Evolution – Test Period")
ax.set_xlabel("Date")
ax.set_ylabel("Portfolio Value")
ax.grid(True)
ax.legend()
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot train performance
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(train_portfolio, label="Portfolio Value (Train)", color='blue')
ax.set_title("Portfolio Evolution – Training Period")
ax.set_xlabel("Date")
ax.set_ylabel("Portfolio Value")
ax.grid(True)
ax.legend()
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()