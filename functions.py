import numpy as np
import pandas as pd

#RSI, MEDIA MOVIL, ICHIMOKUO



def backtesting(capital, t_profit=0.10, t_losses=0.20, n_shares, data, params):
    param1, param2, param3 = params


    # Para posiciones largas
    target_value = long_position * (1+t_profit)
    max_loss = long_position * (1-t_losses)

    # Para posiciones cortas
    target_value_s = short_position * (1+t_profit)
    max_loss_s = short_position * (1-max_loss)

    # Historial de Transacciones
    transacciones = []

    for i in data:
        #Validar SEÑAL AQUI MUEVELE RICHIE, HUGO
        if FIBONACHI >= 30: # FIBONACHI ES DE GAYS
            active_position = {'Date': data.index,
        '   Precio': data[i]}

        #Validar si existe una posicion abierta
        if len(active_position) > 0:
            #LONGS
            long_position = n_shares * data[i]

            if long_position >= target_value:
                capital += long_position
                transacciones.append(active_position)
                active_position = []
            else:
                continue
            if long_position <= max_loss:
                capital += long_position
                transacciones.append(active_position)
                active_position = []
            else:
                continue
            #SHORTS
            short_position = n_shares * active_position.Precio - n_shares * data[i]

            if short_position >= target_value:
                capital += short_position
                transacciones.append(active_position)
                active_position = []
            else:
                continue
            if short_position <= max_loss:
                capital += short_position
                transacciones.append(active_position)
                active_position = []
            else:
                continue
        else:
            continue

