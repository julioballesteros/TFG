import numpy as np
import pandas as pd
import talib
import scipy.stats as scs

from . import data_preparation
from . import stock_metrics as sm


# función para calcular las métricas de la sma
def calculate_sma_metrics(stock_prices, sma_timeperiod):
    stock_prices['sma'] = talib.SMA(stock_prices['close'].values, timeperiod=sma_timeperiod)
    stock_prices['above_sma'] = (stock_prices['close'] >= stock_prices['sma']).astype('int')

    sma_metrics = {'metric': [], 'value': []}

    price_above = stock_prices.above_sma == 1

    # calcular frecuecia relativa de above_sma
    sma_metrics['metric'].append('above_sma_pct')
    sma_metrics['value'].append(sm.series_relative_frequence(stock_prices.above_sma))

    # calcular frecuencia de la tendencia cuando está por encima de above_sma
    sma_metrics['metric'].append('above_sma_bullish_pct')
    sma_metrics['value'].append(sm.bullish_relative_frequence(price_above, stock_prices.future_bullish))

    # calcular frecuencia de la tendencia cuando está por debajo de above_sma
    sma_metrics['metric'].append('below_sma_bullish_pct')
    sma_metrics['value'].append(sm.bullish_relative_frequence(np.logical_not(price_above), stock_prices.future_bullish))

    # calcular p-value para la hipótesis de la sma
    sma_metrics['metric'].append('sma_p-value')
    sma_metrics['value'].append(sm.p_value(
        stock_prices.loc[price_above, 'future_returns'],
        stock_prices.loc[np.logical_not(price_above), 'future_returns']
    ))

    return sma_metrics


# funcion para calcular las métricas del rsi
def calculate_rsi_metrics(stock_prices, rsi_timeperiod):
    stock_prices['rsi'] = talib.RSI(stock_prices['close'].values, timeperiod=rsi_timeperiod)

    rsi_metrics = {'metric': [], 'value': []}

    overbought = stock_prices.rsi >= 70
    oversold = stock_prices.rsi <= 30

    # calcular frecuencia relativa de sobrecompra
    rsi_metrics['metric'].append('overbought_pct')
    rsi_metrics['value'].append(sm.series_relative_frequence(overbought))

    # calcular frecuencia de tendencia en sobrecompra
    rsi_metrics['metric'].append('overbought_bullish_pct')
    rsi_metrics['value'].append(sm.bullish_relative_frequence(overbought, stock_prices.future_bullish))

    # calcular frecuencia relativa de sobreventa
    rsi_metrics['metric'].append('oversold_pct')
    rsi_metrics['value'].append(sm.series_relative_frequence(oversold))

    #calcular frecuencia de tendencia en sobreventa
    rsi_metrics['metric'].append('oversold_bullish_pct')
    rsi_metrics['value'].append(sm.bullish_relative_frequence(oversold, stock_prices.future_bullish))

    # calcular trend change accuracy
    rsi_metrics['metric'].append('trend_change_accuracy')
    rsi_metrics['value'].append(sm.rsi_accuracy(overbought, oversold, stock_prices.future_position))

    # calcular p-value para la hipótesis de sobrecompra
    rsi_metrics['metric'].append('overbought_p-value')
    rsi_metrics['value'].append(sm.p_value(
        stock_prices.loc[overbought, 'future_returns'],
        stock_prices.loc[np.logical_not(overbought), 'future_returns']
    ))

    # calcular p-value para la hipótesis de sobreventa
    rsi_metrics['metric'].append('oversold_p-value')
    rsi_metrics['value'].append(sm.p_value(
        stock_prices.loc[oversold, 'future_returns'],
        stock_prices.loc[np.logical_not(oversold), 'future_returns']
    ))

    return rsi_metrics


# funcion para calcular las métricas del macd
def calculate_macd_metrics(stock_prices, macd_timeperiod):
    stock_prices["macd"], stock_prices["macd_signal"], stock_prices["macd_hist"] = talib.MACD(stock_prices['close'], **macd_timeperiod)

    macd_metrics = {'metric': [], 'value': []}

    buy = stock_prices['macd_hist'] > 0
    above_zero = stock_prices['macd_signal'] >= 0

    # calcular frecuencia de tendencia en pro-trend buy
    macd_metrics['metric'].append('pro-trend_buy_bullish_pct')
    macd_metrics['value'].append(sm.bullish_relative_frequence(buy & above_zero, stock_prices.future_bullish))

    # calcular frecuencia de tendencia en pro-trend sell
    macd_metrics['metric'].append('pro-trend_sell_bullish_pct')
    macd_metrics['value'].append(sm.bullish_relative_frequence(np.logical_not(buy) & np.logical_not(above_zero), stock_prices.future_bullish))

    # calcular frecuencia de tendencia en anti-trend buy
    macd_metrics['metric'].append('anti-trend_buy_bullish_pct')
    macd_metrics['value'].append(sm.bullish_relative_frequence(buy & np.logical_not(above_zero), stock_prices.future_bullish))

    # calcular frecuencia de tendencia en anti-trend sell
    macd_metrics['metric'].append('anti-trend_sell_bullish_pct')
    macd_metrics['value'].append(sm.bullish_relative_frequence(np.logical_not(buy) & above_zero, stock_prices.future_bullish))

    # calcular pro-trend signal accuracy
    macd_metrics['metric'].append('pro-trend_signal_accuracy')
    macd_metrics['value'].append(sm.macd_accuracy(buy, above_zero, stock_prices.future_position))

    # calcular anti-trend signal accuracy
    macd_metrics['metric'].append('anti-trend_signal_accuracy')
    macd_metrics['value'].append(sm.macd_accuracy(buy, np.logical_not(above_zero), stock_prices.future_bullish))

    # calcular p-value para la hipótesis de pro-trend buy
    macd_metrics['metric'].append('pro-trend_buy_p-value')
    macd_metrics['value'].append(sm.p_value(
        stock_prices.loc[buy & above_zero, 'future_returns'],
        stock_prices.loc[np.logical_not(buy & above_zero), 'future_returns']
    ))

    # calcular p-value para la hipótesis de pro-trend sell
    macd_metrics['metric'].append('pro-trend_sell_p-value')
    macd_metrics['value'].append(sm.p_value(
        stock_prices.loc[np.logical_not(buy) & np.logical_not(above_zero), 'future_returns'],
        stock_prices.loc[np.logical_not(np.logical_not(buy) & np.logical_not(above_zero)), 'future_returns']
    ))

    # calcular p-value para la hipótesis de anti-trend buy
    macd_metrics['metric'].append('anti-trend_buy_p-value')
    macd_metrics['value'].append(sm.p_value(
        stock_prices.loc[buy & np.logical_not(above_zero), 'future_returns'],
        stock_prices.loc[np.logical_not(buy & np.logical_not(above_zero)), 'future_returns']
    ))

    # calcular p-value para la hipótesis de anti-trend sell
    macd_metrics['metric'].append('anti-trend_sell_p-value')
    macd_metrics['value'].append(sm.p_value(
        stock_prices.loc[np.logical_not(buy) & above_zero, 'future_returns'],
        stock_prices.loc[np.logical_not(np.logical_not(buy) & above_zero), 'future_returns']
    ))

    return macd_metrics


# funcion para calcular las métricas del ppo
def calculate_ppo_metrics(stock_prices, ppo_timeperiod):
    stock_prices['ppo'] = talib.PPO(stock_prices['close'].values, **ppo_timeperiod)

    ppo_metrics = {'metric': [], 'value': []}

    positive_ppo = stock_prices.ppo >= 0

    # calcular porcentaje relativo del ppo positivo/negativo
    ppo_metrics['metric'].append('positive_ppo_pct')
    ppo_metrics['value'].append(sm.series_relative_frequence(positive_ppo))

    # calcular frecuencia de la tendencia cuando el ppo es positivo
    ppo_metrics['metric'].append('positive_ppo_bullish_pct')
    ppo_metrics['value'].append(sm.bullish_relative_frequence(positive_ppo, stock_prices.future_bullish))

    # calcular frecuencia de la tendencia cuando el ppo es negativo
    ppo_metrics['metric'].append('negative_ppo_bullish_pct')
    ppo_metrics['value'].append(sm.bullish_relative_frequence(np.logical_not(positive_ppo), stock_prices.future_bullish))

    # calcular p-value para la hipótesis del ppo
    ppo_metrics['metric'].append('ppo_p-value')
    ppo_metrics['value'].append(sm.p_value(
        stock_prices.loc[positive_ppo, 'future_returns'],
        stock_prices.loc[np.logical_not(positive_ppo), 'future_returns']
    ))

    return ppo_metrics


# funcion para calcular las métricas de las bbands
def calculate_bbands_metrics(stock_prices, bbands_timeperiod):
    stock_prices['bb_upperband'], stock_prices['bb_middleband'], stock_prices['bb_lowerband'] = talib.BBANDS(stock_prices.close, timeperiod=bbands_timeperiod)

    bbands_metrics = {'metric': [], 'value': []}

    above_bb_upperband = stock_prices.close >= stock_prices.bb_upperband
    below_bb_lowerband = stock_prices.close <= stock_prices.bb_lowerband

    # calcular frecuencia relativa del precio por encima de la banda superior
    bbands_metrics['metric'].append('above_bb_upperband_pct')
    bbands_metrics['value'].append(sm.series_relative_frequence(above_bb_upperband))

    # calcular frecuencia de la tendencia cuando el precio está por encima de la banda superior
    bbands_metrics['metric'].append('above_bb_upperband_bullish_pct')
    bbands_metrics['value'].append(sm.bullish_relative_frequence(above_bb_upperband, stock_prices.future_bullish))

    # calcular frecuencia relativa del precio por debajo de la banda inferior
    bbands_metrics['metric'].append('below_bb_lowerband_pct')
    bbands_metrics['value'].append(sm.series_relative_frequence(below_bb_lowerband))

    # calcular frecuencia de la tendencia cuando el precio está por debajo de la banda inferior
    bbands_metrics['metric'].append('below_bb_lowerband_bullish_pct')
    bbands_metrics['value'].append(sm.bullish_relative_frequence(below_bb_lowerband, stock_prices.future_bullish))

    # calcular p-value para la hipótesis de las bbands
    bbands_metrics['metric'].append('above_bb_upperband_p-value')
    bbands_metrics['value'].append(sm.p_value(
        stock_prices.loc[above_bb_upperband, 'future_returns'],
        stock_prices.loc[stock_prices.close < stock_prices.bb_upperband, 'future_returns']
    ))

    bbands_metrics['metric'].append('below_bb_lowerband_p-value')
    bbands_metrics['value'].append(sm.p_value(
        stock_prices.loc[below_bb_lowerband, 'future_returns'],
        stock_prices.loc[stock_prices.close > stock_prices.bb_lowerband, 'future_returns']
    ))

    return bbands_metrics

# función para calcular las métricas de un horizonte temporal
def calculate_fh_metrics(stock_prices, forecast_horizon, params):
    stock_prices = data_preparation.create_price_change_vars(stock_prices, forecast_horizon)
    stock_prices = data_preparation.create_target_features(stock_prices, forecast_horizon)

    fh_metrics = {'indicator': [], 'parameter': [], 'metric': [], 'value': []}

    # calcular el porcentaje alcista-bajista
    fh_metrics['indicator'].append('general')
    fh_metrics['parameter'].append('NA')
    fh_metrics['metric'].append('bullish_pct')
    fh_metrics['value'].append(sm.series_relative_frequence(stock_prices.bullish))

    position_frequence = stock_prices.position.value_counts() / len(stock_prices.position)

    # calcular el porcentaje de posiciones largas
    fh_metrics['indicator'].append('general')
    fh_metrics['parameter'].append('NA')
    fh_metrics['metric'].append('long_pct')
    fh_metrics['value'].append(position_frequence[1])

    # calcular el porcentaje de posiciones cortas
    fh_metrics['indicator'].append('general')
    fh_metrics['parameter'].append('NA')
    fh_metrics['metric'].append('short_pct')
    fh_metrics['value'].append(position_frequence[0])

    # calcular retornos medios
    fh_metrics['indicator'].append('general')
    fh_metrics['parameter'].append('NA')
    fh_metrics['metric'].append('mean_returns')
    fh_metrics['value'].append(stock_prices.returns.mean())

    # calcular volatilidad media
    fh_metrics['indicator'].append('general')
    fh_metrics['parameter'].append('NA')
    fh_metrics['metric'].append('volatility')
    fh_metrics['value'].append(stock_prices.returns.std())

    # calcular volumen medio
    fh_metrics['indicator'].append('general')
    fh_metrics['parameter'].append('NA')
    fh_metrics['metric'].append('mean_volume')
    fh_metrics['value'].append(stock_prices.volume.mean())

    # calcular p-value para la hipótesis de la diferencia de retornos según tendencia
    fh_metrics['indicator'].append('general')
    fh_metrics['parameter'].append('NA')
    fh_metrics['metric'].append('trend_p-value')
    fh_metrics['value'].append(sm.p_value(
        stock_prices.loc[stock_prices.bullish == 1, 'returns'],
        np.abs(stock_prices.loc[stock_prices.bullish == 0, 'returns'])
    ))

    # calcular p-value para la hipótesis de diferencia de retornos según posición
    fh_metrics['indicator'].append('general')
    fh_metrics['parameter'].append('NA')
    fh_metrics['metric'].append('position_p-value')
    fh_metrics['value'].append(sm.p_value(
        stock_prices.loc[stock_prices.position == 1, 'returns'],
        np.abs(stock_prices.loc[stock_prices.position == 0, 'returns'])
    ))

    # calcular las métricas de cada SMA
    for sma_timeperiod in params['sma_timeperiods']:
        sma_metrics = calculate_sma_metrics(stock_prices, sma_timeperiod)

        fh_metrics['indicator'] += (['sma'] * len(sma_metrics['value']))
        fh_metrics['parameter'] += ([sma_timeperiod] * len(sma_metrics['value']))
        fh_metrics['metric'] += sma_metrics['metric']
        fh_metrics['value'] += sma_metrics['value']

    # calcular las métricas de cada RSI
    for rsi_timeperiod in params['rsi_timeperiods']:
        rsi_metrics = calculate_rsi_metrics(stock_prices, rsi_timeperiod)

        fh_metrics['indicator'] += (['rsi'] * len(rsi_metrics['value']))
        fh_metrics['parameter'] += ([rsi_timeperiod] * len(rsi_metrics['value']))
        fh_metrics['metric'] += rsi_metrics['metric']
        fh_metrics['value'] += rsi_metrics['value']

    # calcular las métricas de cada MACD
    for macd_timeperiod in params['macd_timeperiods']:
        macd_metrics = calculate_macd_metrics(stock_prices, macd_timeperiod)
        #f'{macd_timeperiod["fastperiod"]}-{macd_timeperiod["slowperiod"]}-{macd_timeperiod["signalperiod"]}'

        fh_metrics['indicator'] += (['macd'] * len(macd_metrics['value']))
        fh_metrics['parameter'] += ([macd_timeperiod] * len(macd_metrics['value']))
        fh_metrics['metric'] += macd_metrics['metric']
        fh_metrics['value'] += macd_metrics['value']

    # calcular las métricas de cada PPO
    for ppo_timeperiod in params['ppo_timeperiods']:
        ppo_metrics = calculate_ppo_metrics(stock_prices, ppo_timeperiod)
        #f'{ppo_timeperiod["fastperiod"]}-{ppo_timeperiod["slowperiod"]}'

        fh_metrics['indicator'] += (['ppo'] * len(ppo_metrics['value']))
        fh_metrics['parameter'] += ([ppo_timeperiod] * len(ppo_metrics['value']))
        fh_metrics['metric'] += ppo_metrics['metric']
        fh_metrics['value'] += ppo_metrics['value']

    # calcular las métricas de cada BBands
    for bbands_timeperiod in params['bbands_timeperiods']:
        bbands_metrics = calculate_bbands_metrics(stock_prices, bbands_timeperiod)

        fh_metrics['indicator'] += (['bbands'] * len(bbands_metrics['value']))
        fh_metrics['parameter'] += ([bbands_timeperiod] * len(bbands_metrics['value']))
        fh_metrics['metric'] += bbands_metrics['metric']
        fh_metrics['value'] += bbands_metrics['value']

    return fh_metrics


# función para obtener las métricas de un stock
def calculate_stock_metrics(ticker, start_date, end_date, params):
    stock_prices = data_preparation.get_stock_prices(ticker, start_date, end_date)

    stock_metrics = {'forecast_horizon': [], 'indicator': [], 'parameter': [], 'metric': [], 'value': []}

    for fh in params['forecast_horizons']:
        fh_metrics = calculate_fh_metrics(stock_prices, fh, params)

        stock_metrics['forecast_horizon'] += ([fh] * len(fh_metrics['value']))
        stock_metrics['indicator'] += fh_metrics['indicator']
        stock_metrics['parameter'] += fh_metrics['parameter']
        stock_metrics['metric'] += fh_metrics['metric']
        stock_metrics['value'] += fh_metrics['value']

    return stock_metrics

# función para crear el dataset con todas las métricas
def explore_stocks(tickers, start_date, end_date, params):
    stocks_metrics = {'ticker': [], 'forecast_horizon': [], 'indicator': [], 'parameter': [], 'metric': [], 'value': []}

    for ticker in tickers:
        stock_metrics = calculate_stock_metrics(ticker, start_date, end_date, params)

        stocks_metrics['ticker'] += ([ticker] * len(stock_metrics['value']))
        stocks_metrics['forecast_horizon'] += stock_metrics['forecast_horizon']
        stocks_metrics['indicator'] += stock_metrics['indicator']
        stocks_metrics['parameter'] += stock_metrics['parameter']
        stocks_metrics['metric'] += stock_metrics['metric']
        stocks_metrics['value'] += stock_metrics['value']

    return pd.DataFrame(data=stocks_metrics)
