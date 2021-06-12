import yfinance as yf
import numpy as np
import pandas as pd
import talib

#Función para obtener datos de yfinance
def get_financial_data(ticker_name, start_date, end_date=None):
    ticker = yf.Ticker(ticker_name)
    return ticker.history(start=start_date, end=end_date)

# Función para limpiar los datos
def clean_data(stock_prices):
    stock_prices = stock_prices.loc[:, ['Open', 'High', 'Low', 'Close', 'Volume']]
    stock_prices.columns = ['open', 'high', 'low', 'close', 'volume']
    return stock_prices

# Función para realizar la funcionalidad de esta fase y saltarla en fases posteriores
def get_stock_prices(ticker_name, start_date, end_date=None):
    return clean_data(get_financial_data(ticker_name, start_date=start_date, end_date=end_date))

# Función para eliminar los outliers de los returns del dataframe
def drop_outliers(stock_prices):
    outliers = stock_prices.join(stock_prices.returns.rolling(window=21).agg(['mean', 'std']))
    def indentify_outliers(row, n_sigmas=3):
        x = row['returns']
        mu = row['mean']
        sigma = row['std']

        if (x > mu + n_sigmas*sigma) | (x < mu - n_sigmas*sigma):
            return 1
        else:
            return 0
    outliers['outlier'] = outliers.apply(indentify_outliers, axis=1)
    stock_prices = stock_prices[outliers['outlier'] == 0]

    return stock_prices

# Función para obtener la variable que representa el porcentaje de variación del precio en un horizonte temporal
def get_returns(asset_price, forecast_horizon=1):
    return asset_price.pct_change(forecast_horizon) * 100

# Función para obtener la variable que indica la tendencia de los retornos del activo
def get_bullish(returns):
    return (returns >= 0).astype('int')

# Función para obtener la variable que indica la posicion (long / short / stay) según los retornos
def get_position(returns):
    mean_returns = np.mean(np.abs(returns))
    return np.array([ret >= 0 if np.abs(ret) > mean_returns else 2 for ret in returns], dtype='object').astype('int')

# Función para realizar las modificaciones de esta fase y saltarla en fases posteriores
def create_price_change_vars(stock_prices, forecast_horizon=1):
    stock_prices['returns'] = get_returns(stock_prices.close, forecast_horizon)
    stock_prices['bullish'] = get_bullish(stock_prices.returns)
    stock_prices['position'] = get_position(stock_prices.returns)

    stock_prices = drop_outliers(stock_prices)

    return stock_prices

# Función para obtener las variables objetivo
def create_target_features(stock_prices, forecast_horizon):

    # crear el target precio de cierre futuro (future_close)
    stock_prices['future_close'] = stock_prices['close'].shift(-forecast_horizon)

    # crear el target cambio porcentual futuro (future_returns)
    stock_prices['future_returns'] = get_returns(stock_prices.close, forecast_horizon).shift(-forecast_horizon)

    # crear el target tendencia futura (future_bullish)
    stock_prices['future_bullish'] = get_bullish(stock_prices['future_returns'])

    # crear el target posicion futura (future_position)
    stock_prices['future_position'] = get_position(stock_prices['future_returns'])

    stock_prices = drop_outliers(stock_prices)

    return stock_prices

# Función para obtener el indicador SMAD (SMA Distance)
# indicador = (precio - media) / media
def SMAD(close, timeperiod=14):
    sma = talib.SMA(close, timeperiod)
    return 100 * (close - sma) / sma

# Función para obtener el indicador BBD (BBands Distance)
# indicador = (precio - bb_middleband) / (bb_upperband - bb_middleband) = distancia a la media / amplitud de las bandas
def BBD(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0):
    bb_upperband, bb_middleband, bb_lowerband = talib.BBANDS(close, timeperiod, nbdevup, nbdevdn, matype)
    return (close - bb_middleband) / (bb_upperband - bb_middleband)
