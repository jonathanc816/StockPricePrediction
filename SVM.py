from sklearn.svm import SVC
from sklearn import svm
  
import pandas as pd
import numpy as np

import matplotlib
# matplotlib.use('GTK3Agg')
import matplotlib.pyplot as plt
# plt.style.use('seaborn-darkgrid')


def simple_SVM():
    df = pd.read_csv('data/AAPL.csv')

    # predictor vars (feature vectors X)
    df['Open-Close'] = df["Open"] - df["Close"]
    df['High-Low'] = df["High"] - df["Low"]
    X = df[['Open-Close', 'High-Low']] 
    # print(X.head())
    
    # target vars (feature vectors Y)
    y = np.where(df['Close'].shift(-1) > df['Close'], 1, -1)
    
    split = int(0.8 * len(df)) 
    
    X_train = X[:split]
    y_train = y[:split]
    X_test = X[split:]
    y_test = y[split:]

    # fit our SVM model and calculate its score 
    svm_model = svm.SVC(kernel='rbf')
    svm_model.fit(X_train, y_train)
    # mean accuracy on the given test data and labels
    score = svm_model.score(X_test, y_test) 
    print("mean acc =",score)

    prediction = svm_model.predict(X_test)
    df['Predicted_Signal'] = svm_model.predict(X)
    df['pct_change'] = df["Close"].pct_change() 
    df['strat_change'] = df["pct_change"] * df["Predicted_Signal"].shift(1)
    df['cumulative_change'] = df['pct_change'].cumsum()
    df['cumulative_strat'] = df['strat_change'].cumsum()

    plt.figure()
    plt.plot(df['cumulative_change'], label="actual returns")
    plt.plot(df['cumulative_strat'], label='predicted returns')
    plt.legend()


    # plt.figure()
    # plt.plot(y_test, label='actual')
    # plt.plot(prediction, label='predicted')
    # plt.legend()
    plt.show()


def main():
    simple_SVM()


if __name__ == "__main__": 
	main()