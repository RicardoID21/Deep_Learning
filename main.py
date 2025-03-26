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
    return result["sharpe"]

study = optuna.create_study(direction="maximize")
study.optimize(wrapped_objective, n_trials=50)

# Mostrar mejores resultados
print("Mejor Sharpe:", study.best_value)
print("Mejores parámetros:", study.best_params)

# Ver todas las métricas del mejor resultado
best_index = [i for i, r in enumerate(results) if r["sharpe"] == study.best_value][0]
best_result = results[best_index]
print("Todas las métricas del mejor resultado:")
print(best_result)