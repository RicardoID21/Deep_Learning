import numpy as np
import optuna
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from functions import objective_func, evaluate_with_params

# Carga de Datos
url_train = "aapl_5m_train.csv"
train = pd.read_csv(url_train)

url_test = "aapl_5m_test.csv"
test = pd.read_csv(url_test)

train = train[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]
train.set_index(train.columns[0], inplace=True)

test = test[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]
test.set_index(test.columns[0], inplace=True)

# Correr el estudio (optimizamos Sharpe ratio)
results = []

def wrapped_objective(trial):
    result = objective_func(trial, train)
    results.append(result)
    return result["sharpe"] if not np.isnan(result["sharpe"]) else -np.inf  # Protección por si da NaN

study = optuna.create_study(direction="maximize")
study.optimize(wrapped_objective, n_trials=50)

# Mostrar resultados finales
print("\n🔍 Mejor Sharpe:", round(study.best_value, 4))
print("⚙️  Mejores parámetros:", study.best_params)

# Extraer métricas del mejor resultado
best_index = max(
    range(len(results)),
    key=lambda i: results[i]["sharpe"] if not np.isnan(results[i]["sharpe"]) else -np.inf
)
best_result = results[best_index]

print("\n📊 Métricas del mejor resultado:")
for k, v in best_result.items():
    if isinstance(v, float):
        print(f"{k.capitalize():<25}: {round(v, 4)}")
    else:
        print(f"{k.capitalize():<25}: {v}")


train_result = evaluate_with_params(study.best_params, train)
test_result = evaluate_with_params(study.best_params, test)

test_portfolio = pd.Series(test_result['return'], index=test.iloc[-len(test_result['return']):].index)
train_portfolio = pd.Series(train_result['return'], index=train.iloc[-len(train_result['return']):].index)

test_portfolio.index = pd.to_datetime(test_portfolio.index)
train_portfolio.index = pd.to_datetime(train_portfolio.index)

# Formatear los ejes de fecha
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(test_portfolio, label="Portfolio Value (Test)", color='black')
ax.set_title("Evolución del portafolio en Test")
ax.set_xlabel("Fecha")
ax.set_ylabel("Valor del portafolio")
ax.grid(True)
ax.legend()

# Mostrar un tick cada día o cada X horas
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))  # cada 2 días
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Asegúrate de que el índice sea datetime
train_portfolio.index = pd.to_datetime(train_portfolio.index)

# Crear gráfico con formateo de fechas
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(train_portfolio, label="Portfolio Value (Train)", color='blue')
ax.set_title("Evolución del portafolio en Entrenamiento")
ax.set_xlabel("Fecha")
ax.set_ylabel("Valor del portafolio")
ax.grid(True)
ax.legend()

# Mostrar un tick cada 2 días
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

# Rotar etiquetas y ajustar diseño
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


print("\n📉 Resultados en TEST:")
for k, v in test_result.items():
    print(f"{k.capitalize():<25}: {round(v, 4) if isinstance(v, float) else v}")







