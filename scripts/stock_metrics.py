import numpy as np
import pandas as pd
import talib
import scipy.stats as scs

from . import data_preparation


# función para obtener el p-value para una hipótesis de diferencia entre distribuciones
# se utiliza la prueba Mann-Whitney U
# cada muestra debe tener más de 20 observaciones
def p_value(sample1, sample2):
    if len(sample1) > 20 and len(sample2) > 20:
        return scs.mannwhitneyu(sample1, sample2)[1]

    return np.nan


# calcula  la frecuencia relativa del valor 1 en una serie booleana
def series_relative_frequence(series):

    # la serie no puede estar vacia
    if len(series) == 0:
        return np.nan

    return series.sum() / len(series)


# calcula la frecuencia relativa de la tendencia alcista indexada por una serie booleana
def bullish_relative_frequence(series, future_bullish):

    # tiene que haber valores iguales al indicado en la serie
    if np.sum(series) == 0:
        return np.nan

    return series_relative_frequence(future_bullish[series])


def rsi_accuracy(overbought, oversold, future_position):

    position_count = (future_position[overbought] != 2).sum() + (future_position[oversold] != 2).sum()

    #si no hay valores en sobrecompra y sobreventa con posiciones no puede calcularse
    if position_count == 0:
        return np.nan

    position_overbought = future_position[overbought].value_counts()
    position_oversold = future_position[oversold].value_counts()

    overbought_sort = position_overbought[0] if 0 in position_overbought.index else 0
    oversold_long = position_oversold[1] if 1 in position_oversold.index else 0

    return (overbought_sort + oversold_long) / position_count


def macd_accuracy(buy, above_zero, future_position):

    position_count = (future_position[buy & above_zero] != 2).sum() + (future_position[(~buy) & (~above_zero)] != 2).sum()

    # si no hay valores en zonas pro trend no se calcula
    if position_count == 0:
        return np.nan

    position_buy = future_position[buy & above_zero].value_counts()
    position_sell = future_position[np.logical_not(buy) & np.logical_not(above_zero)].value_counts()

    buy_long = position_buy[1] if 1 in position_buy.index else 0
    sell_sort = position_sell[0] if 0 in position_sell.index else 0

    return (buy_long + sell_sort) / position_count
