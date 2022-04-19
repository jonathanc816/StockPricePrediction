import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import yfinance as yf
from sklearn import preprocessing
from datetime import datetime
import matplotlib.dates as md
from torch.utils.data import Dataset


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


scaler = preprocessing.StandardScaler()
def standardize(col):
    fitting = scaler.fit(np.array(col).reshape(-1, 1))
    x_scaled = fitting.transform(np.array(col).reshape(-1, 1))
    return x_scaled, scaler

def split_data(df, duration):
    input_x = []
    target = []
    for i in range(duration, len(df)):
        input_x.append(df[i - duration:i, 0])
        target.append(df[i, 0])
    input_x, target = np.array(input_x), np.array(target)
    return input_x, target

#APPL
appl_price = aapl_csv_df['Close']
aapl_close = standardize(aapl_csv_df['Close'])[0]
aapl_close_scaler = standardize(aapl_csv_df['Close'])[1]
aapl_training_set = aapl_close[:1600] # 75% training set
aapl_test_set = aapl_close[1600:] # 25% test set

aapl_training_x, aapl_training_y = split_data(aapl_training_set, 24)


aapl_test_x, aapl_test_y = split_data(aapl_test_set, 24)


class StockData(Dataset):
    def __init__(self, x, y):
        x = np.reshape(x, (x.shape[0], x.shape[1], 1))
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return (self.x[index], self.y[index])


training_set = StockData(aapl_training_x, aapl_training_y)
test_set = StockData(aapl_test_x, aapl_test_y)
print("Training data shape", training_set.x.shape, training_set.y.shape)
print("Test data shape", test_set.x.shape, test_set.y.shape)
