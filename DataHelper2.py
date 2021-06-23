import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

import Util.MathUtil as MathUtil
from Util.MathUtil import fractional_difference, dwt_denoise, get_garch_model
from Util.DBHelper import DBHelper

###
# Data helper 2 class is to load the finance data, calculate all indicators
###
class DataHelper2:
    URL_AGGREGATES = 'https://api.polygon.io/v2/aggs/ticker/{}/range/1/minute/{}/{}?unadjusted=true&sort=asc&apiKey={}'
    FORCE_UPDATE = True

    def __init__(self):
        self.data = None
        # self.num_days = (datetime.strptime(date_end, '%Y-%m-%d') - datetime.strptime(date_start, '%Y-%m-%d')).days
        # self.symbols = initial_symbols
        self.api_key = open('polygon.key', 'r').read().strip()
        self.dbHelper = DBHelper()

    # Pickle dump method
    def __getstate__(self):
        new_dict = self.__dict__.copy()
        del new_dict['dbHelper']
        return new_dict

    # Pickle load method
    def __setstate__(self, new_state):
        self.__dict__.update(new_state)
        self.dbHelper = DBHelper()

    def compute_default_indicators(self, data):
        data['returns'] = data['close'] - data['close'].shift(1)
        data['log_close'] = np.log(data['close'])
        # data['log_close'] = data['log_close'] - data['log_close'].iloc[0]
        data['pct_close'] = np.cumsum((data['close'] - data['close'].shift(1)) / data['close'] * 100).dropna()
        fd_close = MathUtil.fractional_difference(data['close'], 0.45)
        data['fd_close'] = fd_close
        # _data['time_of_day'] = np.linspace(0, 1, len(_data))
        return data

    # DB loader includes first day, excludes last day
    def load_ohlc_db(self, symbol, date_start, date_end):
        data = self.dbHelper.get_ohlc(symbol, 'minute', date_start, date_end)
        return data

    # This is going to error if no data is retreived
    # Polygon includes data for both dates provided
    def load_ohlc_polygon(self, symbol, date_start, date_end):
        date_cur = datetime.strptime(date_start, '%Y-%m-%d')
        date_end = datetime.strptime(date_end, '%Y-%m-%d')
        frames = []
        while date_cur <= date_end:
            date = date_cur.strftime('%Y-%m-%d')
            url = DataHelper2.URL_AGGREGATES.format(symbol, date, date, self.api_key)
            response = requests.get(url).json()

            if 'results' in response:
                # raise Exception('Error retreiving data for symbol: {}'.format(str(symbol)))
            
                raw_data = response['results']

                _data = pd.DataFrame(raw_data, columns=['o', 'h', 'l', 'c', 'v', 't'])
                _data = _data.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume', 't': 'time'})
                _data = _data.set_index('time')
                # _data = self.compute_default_indicators(_data)

                frames.append(_data)

            date_cur = date_cur + timedelta(days=1)
        data = pd.concat(frames)
        
        return data

    def has_symbol(self, symbol):
        return len(self.data[self.data['symbol'] == symbol]) > 0
        
    def indicator_to_column(self, indicator):
        if isinstance(indicator, list):
            if isinstance(indicator[1], list):
                return '{}_{}'.format(indicator[0], '_'.join([str(x) for x in indicator[1]]))
            else:
                return '{}_{}'.format(indicator[0], indicator[1])
        else:
            result = ''
            if 'period' in indicator and indicator['period'] is not None:
                result = self.indicators_to_columns([[indicator['type'], indicator['period']]])[0]
            else:
                result = indicator['type']

            gradient = 'gradient' in indicator and indicator['gradient']
            sign = 'sign' in indicator and indicator['sign']
            prefixes = ''
            if gradient: prefixes = 'Δ' + prefixes
            if sign: prefixes = '±' + prefixes
            if not prefixes == '': result = '{}_{}'.format(prefixes, result)
            return result
    
    def data_length(self, symbol):
        return len(self.data[symbol])

    # update this for new dict based indicators
    def indicators_to_columns(self, indicator_list):
        columns = []
        for indicator in indicator_list:
            columns.append(self.indicator_to_column(indicator))
        return columns

    # This takes a dict
    def calculate_indicator(self, symbol, indicator):
        key = self.indicator_to_column(indicator)

        if key in self.data[symbol].columns:
            return

        high = self.data[symbol]['high']
        low = self.data[symbol]['low']
        close = self.data[symbol]['close']
        volume = self.data[symbol]['volume']
        gradient = 'gradient' in indicator and indicator['gradient']
        sign = 'sign' in indicator and indicator['sign']
        
        if indicator['type'] == 'returns':
            result = close - close.shift(1)
        if indicator['type'] == 'ma':
            result = MathUtil.indicator_sma(close, indicator['period'], gradient=gradient, sign=sign)
        elif indicator['type'] == 'rsi':
            result = MathUtil.indicator_rsi(close, indicator['period'], gradient=gradient, sign=sign)
        elif indicator['type'] == 'vwap':
            result = MathUtil.indicator_vwap(high, low, close, volume, indicator['period'], gradient=gradient, sign=sign)
        elif indicator['type'] == 'cci':
            result = MathUtil.indicator_cci(high, low, close, indicator['period'], gradient=gradient, sign=sign)
        elif indicator['type'] == 'macd':
            result = MathUtil.indicator_macd(close, indicator['period'], gradient=gradient, sign=sign)
        elif indicator['type'] == 'atr':
            result = MathUtil.indicator_atr(high, low, close, indicator['period'], gradient=gradient, sign=sign)
        elif indicator['type'] == 'prsi':
            result = MathUtil.indicator_premier_rsi(high, low, close, indicator['period'], gradient=gradient, sign=sign)
        elif indicator['type'] == 'sqzmom':
            result = MathUtil.indicator_squeeze_momentum(high, low, close, gradient=gradient, sign=sign)
        else:
            print('DataHelper, Indicator is not available: ' + str(indicator))
            
        self.data[symbol][key] = result

    def calculate_indicators(self, symbols, indicators):
        for symbol in symbols:
            # self.compute_default_indicators(_data)
            for indicator in indicators:
                self.calculate_indicator(symbol, indicator)

    def get_data(self, symbols, date_start, date_end, indicators={}):
        # results = []
        for symbol in symbols:
            if self.data is None or symbol not in self.data:
                try:
                    results = self.load_ohlc_db(symbol, date_start, date_end)
                    if len(results) < 1 or DataHelper2.FORCe_UPDATE:
                        print('loading data from polygon')
                        results = self.load_ohlc_polygon(symbol, date_start, date_end)
                        self.dbHelper.insert_ohlc('minute', symbol, results)

                    # results['symbol'] = symbol
                    # if self.data is None:
                    #     self.data = results
                    # else:
                    #     self.data = pd.concat([self.data, results])
                except Exception as err:
                    print('loading data from polygon')
                    results = self.load_ohlc_polygon(symbol, date_start, date_end)
                    self.dbHelper.insert_ohlc('minute', symbol, results)

                return results

        # TODO make these transforms toggleable
        # for symbol in symbols:
        #     self.data[symbol] = symbol.reset_index().drop('time', axis=1)
        self.calculate_indicators(symbols, indicators)
        return self.data