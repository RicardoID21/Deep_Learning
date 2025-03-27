from functions import objective_func
import pandas as pd
import optuna
import numpy as np

# Carga de Datos
url_train = "aapl_5m_train.csv"
train = pd.read_csv(url_train)

url_test = "aapl_5m_test.csv"
test = pd.read_csv(url_test)

train = train[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]
train.set_index(train.columns[0], inplace=True)

# Correr el estudio (optimizamos Sharpe ratio)
results = []

def wrapped_objective(trial):
    result = objective_func(trial, train)
    results.append(result)
    return result["sharpe"] if not np.isnan(result["sharpe"]) else -np.inf  # Protección por si da NaN

study = optuna.create_study(direction="maximize")
study.optimize(wrapped_objective, n_trials=50)

# Mostrar resultados finales
print("\n🔍 Mejor Sharpe:", round(study.best_value * np.sqrt(19656), 4))
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