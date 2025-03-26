import numpy as np
import pandas as pd
#RSI, MEDIA MOVIL, ICHIMOKUO



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

