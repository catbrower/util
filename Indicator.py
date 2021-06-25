import numpy as np

from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volume import VolumeWeightedAveragePrice
from ta.trend import CCIIndicator, EMAIndicator
from ta.volatility import AverageTrueRange, KeltnerChannel, BollingerBands

from util.MathUtil import fast_sigmoid

def apply_transforms(values, gradient, sign):
    result = values
    if gradient:
        result = np.gradient(result)

    if sign:
        result = [np.sign(x) for x in result]

    return result

def base_data(data):
    return data['open'], data['high'], data['low'], data['close'], data['volume']

# Do not pass a dataframe to this
def moving_average(data, period, gradient=False, sign=False):
    result = data.rolling(int(period)).mean()
    return apply_transforms(result, gradient, sign)

def indicator_macd(data, periods, gradient=False, sign=False):
    _, _, _, close, _ = base_data(data)
    ma_short = close.rolling(periods[0]).mean()
    ma_long = close.rolling(periods[1]).mean()

    result = fast_sigmoid((ma_long - ma_short) / ma_long  * 100)
    return apply_transforms(result, gradient, sign)

def indicator_rsi(data, period, gradient=False, sign=False):
    _, _, _, close, _ = base_data(data)
    rsi = RSIIndicator(close, period)._rsi
    result = rsi / 100.0 - 0.5
    return apply_transforms(result, gradient, sign)

def indicator_sma(data, period, gradient=False, sign=False):
    _, _, _, close, _ = base_data(data)
    result = close.rolling(period).mean()
    return apply_transforms(result, gradient, sign)

def indicator_vwap(data, period, gradient=False, sign=False):
    _, high, low, close, volume = base_data(data)
    vwap = VolumeWeightedAveragePrice(high, low, close, volume, period).vwap
    result = fast_sigmoid((vwap - close) / close * 100.0)
    return apply_transforms(result, gradient, sign)

def indicator_cci(data, period, gradient=False, sign=False):
    _, high, low, close, _ = base_data(data)
    cci = CCIIndicator(high, low, close, period)._cci
    result = fast_sigmoid(cci / 100.0)
    return apply_transforms(result, gradient, sign)

def indicator_atr(data, period, gradient=False, sign=False):
    _, high, low, close, _ = base_data(data)
    result = AverageTrueRange(high, low, close, window=period)._atr
    return apply_transforms(result, gradient, sign)

def indicator_premier_rsi(data, period, stochlen=8, smoothlen=25, gradient=False, sign=False):
    _, _, _, close, _ = base_data(data)
    r = RSIIndicator(close, period).rsi()
    sk = StochasticOscillator(r, r, r, stochlen).stoch()
    llen = round(np.sqrt(smoothlen))
    nsk = 0.1 * (sk - 50)
    ss = EMAIndicator(EMAIndicator(nsk, llen).ema_indicator(), llen).ema_indicator()
    expss = np.exp(ss)
    result = (expss -1) / (expss + 1)
    return apply_transforms(result, gradient, sign)

def indicator_squeeze_momentum(data, bbPeriod=20, bbMult=2, kcPeriod=20, kcMult=1.5, useTrueRange=True, gradient=False, sign=False):
    _, high, low, close, _ = base_data(data)
    
    #sqzOn = 2, noSqz = 1, sqzOff = 0
    def getSqueeze(values):
        lowerBB = values[0]
        upperBB = values[1]
        lowerKC = values[2]
        upperKC = values[3]
        if lowerBB > lowerKC and upperBB < upperKC:
            return 2
        elif lowerBB < lowerKC and upperBB > upperKC:
            return 0
        else:
            return 1
    
    # Calculate BB
    source = close
    basis = source.rolling(bbPeriod).mean()

    # dev = multKC * np.std(source[-bbPeriod:])
    # upperBB = basis + dev
    # lowerBB = basis - dev
    BB = BollingerBands(close, bbPeriod, window_dev=bbMult)
    upperBB = BB.bollinger_hband()
    lowerBB = BB.bollinger_lband()

    # Calculate KC, tr represents true range
    KC = KeltnerChannel(high, low, close, kcPeriod)
    upperKC = KC.keltner_channel_hband()
    lowerKC = KC.keltner_channel_lband()

    indicators = list(zip(lowerBB, upperBB, lowerKC, upperKC))
    squeeze = list(map(getSqueeze, indicators))

    # linreg(source, length, offset)
    avg1 = (high.rolling(kcPeriod).apply(lambda x: max(x)) + low.rolling(kcPeriod).apply(lambda x: min(x))) / 2
    avg2 = (avg1 + close.rolling(kcPeriod).mean()) / 2
    # val = linear_model.LinearRegression().fit(source - avg2, lengthKC, 0)
    result = (source - avg2).rolling(kcPeriod).apply(lambda x: np.polyfit(x.index, x.values, 1)[0])
    # sqzOn = [x for x in range(len(squeeze)) if squeeze[x] == 2]
    return result

class IndicatorFactory:
    def build(name, period, gradient=False, sign=False):
        indicator = Indicator(name, period)

        if name == 'macd':
            indicator.eval = indicator_macd
        elif name == 'rsi':
            indicator.eval = indicator_rsi
        elif name == 'sma':
            indicator.eval = indicator_sma
        elif name == 'vwap':
            indicator.eval = indicator_vwap
        elif name == 'cci':
            indicator.eval = indicator_cci
        elif name == 'atr':
            indicator.eval = indicator_atr
        elif name == 'prsi':
            indicator.eval = indicator_premier_rsi
        elif name == 'sqzmom':
            indicator.eval = indicator_squeeze_momentum
        else:
            raise Exception('Unknown indicator: {}'.formate(name))
        return indicator

class Indicator:
    def __init__(self, name, period, gradient=False, sign=False):
        self.name = name
        self.period = period
        self.gradient = gradient
        self.sign = sign

    def eval(self, data):
        raise Exception('Function not implemented')

    def to_string(self):
        parts = [self.name, self.period]
        if self.gradient:
            parts += 'dx'

        if self.sign:
            parts += '+/-'

        return '_'.join(parts)