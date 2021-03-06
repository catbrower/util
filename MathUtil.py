import numpy as np
import pandas as pd
import pywt

from datetime import datetime, timedelta

from nolitsa import surrogates
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from arch import arch_model

def add_days(start_date, num_days):
    date_format = '%Y-%m-%d'
    new_date = datetime.strptime(start_date, date_format) + timedelta(days=num_days)
    return new_date.strftime(date_format)
    
def flatten_list(values):
    result = []
    for x in values:
        if isinstance(x, list):
            result += flatten_list(x)
        else:
            result.append(x)
    return result

def fractional_difference(series, order, cutoff=1e-4):
    def get_weights(d, lags):
        weights = [1]
        for k in range(1, lags):
            weights.append(-weights[-1] * ((d - k + 1)) / k)
        weights = np.array(weights).reshape(-1, 1)
        return weights

    def find_cutoff(cutoff_order, cutoff, start_lags):
        val = np.inf
        lags = start_lags
        while abs(val) > cutoff:
            weight = get_weights(cutoff_order, lags)
            val = weight[len(weight) - 1]
            lags += 1
        return lags

    lag_cutoff = ( find_cutoff(order, cutoff, 1) )
    weights = get_weights(order, lag_cutoff)
    result = 0
    for k in range(lag_cutoff):
        result += weights[k] * series.shift(k).fillna(0)

    return result[lag_cutoff:]

def dwt_denoise(data):
    #len(c)%(2**n)=0; where n = level; I used len(c)=512
    coeffs = pywt.wavedec(data, 'db4', level=5) #returns array of cA5,cD5,cD4,cD3,...
    for i in range(1, len(coeffs)):
        temp = coeffs[i]
        mu = temp.mean()
        sigma = temp.std()
        omega = temp.max()
        kappa = (omega - mu) / sigma  #threshold value
        coeffs[i] = pywt.threshold(temp, kappa, mode='garrote')

    return pywt.waverec(coeffs, 'db4')

def cwt_spectrum(data):
    scales = np.arange(1, len(data) + 1)
    img = pywt.cwt(data, scales, 'morl')
    return img[0]

# Make sure there is no na
def aaft(returns):
    return surrogates.aaft(returns).cumsum()

def get_arima_forecast(data):
  model = ARIMA(data['dwt'], order=(5, 0, 0)).fit()
  residuals = pd.DataFrame(model.resid)
  # print(model.summary())
  print('-' * 25)
  print(model.forecast())
  residuals.plot()
  # residuals.plot(kind='kde')

  # Do predictions from the ARIMA
  # Maybe just do this with prophet?
  split_index = int(len(data) * 2/3)
  data_train = data[0:split_index]
  data_eval = data[split_index:]
  predictions = []

  # for i in range(len(data_eval)):
  #   model = ARIMA(data_train + predictions, order=(5, 0, 0)).fit()
  #   yhat = model.forecast()[0]
  #   predictions.append(yhat)

def get_garch_model(data, forecast_length=10):
  # seed(1)
  # These p and q value are probably wrong
  model = arch_model(data).fit(update_freq=5)
  return (model.conditional_volatility, model.forecast(horizon=forecast_length).variance.values)

def plot_acf_grpahs(data):
  squared_data = np.array([x**2 for x in data])
  plot_acf(squared_data)
  plot_pacf(squared_data)

def func_relu(x):
    return x if x > 0 else 0

def func_randomize(x, scale=1):
    return x + np.random.normal() * scale / 2

def sigmoid(x):
    if x > 100:
        return 1
    if x < -100:
        return 0
    return 1 / (1 + np.exp(-x))

def fast_sigmoid(x):
    return x / (1 + abs(x))

###
# Modwt
# https://github.com/pistonly/modwtpy
###

def upArrow_op(li, j):
    if j == 0:
        return [1]
    N = len(li)
    li_n = np.zeros(2 ** (j - 1) * (N - 1) + 1)
    for i in range(N):
        li_n[2 ** (j - 1) * i] = li[i]
    return li_n

def period_list(li, N):
    n = len(li)
    # append [0 0 ...]
    n_app = N - np.mod(n, N)
    li = list(li)
    li = li + [0] * n_app
    if len(li) < 2 * N:
        return np.array(li)
    else:
        li = np.array(li)
        li = np.reshape(li, [-1, N])
        li = np.sum(li, axis=0)
        return li

def circular_convolve_mra(h_j_o, w_j):
    ''' calculate the mra D_j'''
    N = len(w_j)
    l = np.arange(N)
    D_j = np.zeros(N)
    for t in range(N):
        index = np.mod(t + l, N)
        w_j_p = np.array([w_j[ind] for ind in index])
        D_j[t] = (np.array(h_j_o) * w_j_p).sum()
    return D_j

def circular_convolve_d(h_t, v_j_1, j):
    '''
    jth level decomposition
    h_t: \tilde{h} = h / sqrt(2)
    v_j_1: v_{j-1}, the (j-1)th scale coefficients
    return: w_j (or v_j)
    '''
    N = len(v_j_1)
    L = len(h_t)
    w_j = np.zeros(N)
    l = np.arange(L)
    for t in range(N):
        index = np.mod(t - 2 ** (j - 1) * l, N)
        v_p = np.array([v_j_1[ind] for ind in index])
        w_j[t] = (np.array(h_t) * v_p).sum()
    return w_j

def circular_convolve_s(h_t, g_t, w_j, v_j, j):
    '''
    (j-1)th level synthesis from w_j, w_j
    see function circular_convolve_d
    '''
    N = len(v_j)
    L = len(h_t)
    v_j_1 = np.zeros(N)
    l = np.arange(L)
    for t in range(N):
        index = np.mod(t + 2 ** (j - 1) * l, N)
        w_p = np.array([w_j[ind] for ind in index])
        v_p = np.array([v_j[ind] for ind in index])
        v_j_1[t] = (np.array(h_t) * w_p).sum()
        v_j_1[t] = v_j_1[t] + (np.array(g_t) * v_p).sum()
    return v_j_1

def modwt(x, filters, level):
    '''
    filters: 'db1', 'db2', 'haar', ...
    return: see matlab
    '''
    # filter
    wavelet = pywt.Wavelet(filters)
    h = wavelet.dec_hi
    g = wavelet.dec_lo
    h_t = np.array(h) / np.sqrt(2)
    g_t = np.array(g) / np.sqrt(2)
    wavecoeff = []
    v_j_1 = x
    for j in range(level):
        w = circular_convolve_d(h_t, v_j_1, j + 1)
        v_j_1 = circular_convolve_d(g_t, v_j_1, j + 1)
        wavecoeff.append(w)
    wavecoeff.append(v_j_1)
    return np.vstack(wavecoeff)

def imodwt(w, filters):
    ''' inverse modwt '''
    # filter
    wavelet = pywt.Wavelet(filters)
    h = wavelet.dec_hi
    g = wavelet.dec_lo
    h_t = np.array(h) / np.sqrt(2)
    g_t = np.array(g) / np.sqrt(2)
    level = len(w) - 1
    v_j = w[-1]
    for jp in range(level):
        j = level - jp - 1
        v_j = circular_convolve_s(h_t, g_t, w[j], v_j, j + 1)
    return v_j

def modwtmra(w, filters):
    ''' Multiresolution analysis based on MODWT'''
    # filter
    wavelet = pywt.Wavelet(filters)
    h = wavelet.dec_hi
    g = wavelet.dec_lo
    # D
    level, N = w.shape
    level = level - 1
    D = []
    g_j_part = [1]
    for j in range(level):
        # g_j_part
        g_j_up = upArrow_op(g, j)
        g_j_part = np.convolve(g_j_part, g_j_up)
        # h_j_o
        h_j_up = upArrow_op(h, j + 1)
        h_j = np.convolve(g_j_part, h_j_up)
        h_j_t = h_j / (2 ** ((j + 1) / 2.))
        if j == 0: h_j_t = h / np.sqrt(2)
        h_j_t_o = period_list(h_j_t, N)
        D.append(circular_convolve_mra(h_j_t_o, w[j]))
    # S
    j = level - 1
    g_j_up = upArrow_op(g, j + 1)
    g_j = np.convolve(g_j_part, g_j_up)
    g_j_t = g_j / (2 ** ((j + 1) / 2.))
    g_j_t_o = period_list(g_j_t, N)
    S = circular_convolve_mra(g_j_t_o, w[-1])
    D.append(S)
    return np.vstack(D)