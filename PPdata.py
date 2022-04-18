import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import yfinance as yf
from sklearn import preprocessing
import torch as tf
from datetime import datetime
import matplotlib.dates as md

#aapl_df = yf.download('AAPL', start='2013-11-05', end='2022-04-13', progress=False)
aapl_csv_df = pd.read_csv('data/AAPL.csv')
aapl_time = [datetime.fromisoformat(i) for i in aapl_csv_df['Date']]
plt.figure()
formatter = md.DateFormatter("%Y")
locator = md.YearLocator()
ax = plt.gca()
ax.xaxis.set_major_formatter(formatter)
ax.xaxis.set_major_locator(locator)
plt.plot(aapl_time, aapl_csv_df['Close'])
plt.title("aapl 2013-2022")
plt.xlabel('year')
plt.ylabel('closing price')

sp500_csv_df = pd.read_csv('data/s&p500.csv')
sp500_time = [datetime.strptime(i, "%Y-%m-%dT%H:%M:%SZ") for i in sp500_csv_df['time']]
plt.figure()
formatter = md.DateFormatter("%Y")
locator = md.YearLocator()
ax = plt.gca()
ax.xaxis.set_major_formatter(formatter)
ax.xaxis.set_major_locator(locator)
plt.plot(sp500_time, sp500_csv_df['close'])
plt.title("s&p500 2013-2022")
plt.xlabel('year')
plt.ylabel('closing price')


def standardize(col):
    scaler = preprocessing.StandardScaler().fit(np.array(col).reshape(-1, 1))
    x_scaled = scaler.transform(np.array(col).reshape(-1, 1))
    return x_scaled

def split_data(df, duration):
    input_x = []
    target = []
    for i in range(duration, len(df)):
        input_x.append(df[i - duration:i, 0])
        target.append(df[i, 0])
    input_x, target = np.array(input_x), np.array(target)
    return input_x, target


aapl_close = standardize(aapl_csv_df['Close'])
aapl_training_set = aapl_close[:2000]
aapl_test_set = aapl_close[2000:]

aapl_training_x, aapl_training_y = split_data(aapl_training_set, 24)
aapl_training_x = tf.from_numpy(aapl_training_x).type(tf.Tensor)
aapl_training_y = tf.from_numpy(aapl_training_y).type(tf.Tensor)
aapl_training_x = aapl_training_x.unsqueeze(2)
# aapl_training_x = np.reshape(aapl_training_x, (aapl_training_x.shape[0],
#                                                aapl_training_x.shape[1], 1))

aapl_test_x, aapl_test_y = split_data(aapl_test_set, 24)
aapl_test_x = tf.from_numpy(aapl_test_x).type(tf.Tensor)
aapl_test_y = tf.from_numpy(aapl_test_y).type(tf.Tensor)
aapl_test_x = aapl_test_x.unsqueeze(2)
# aapl_test_x = np.reshape(aapl_test_x, (aapl_test_x.shape[0],aapl_test_x.shape[1], 1))

