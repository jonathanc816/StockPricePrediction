from sklearn.svm import SVC
from sklearn import svm
  
import pandas as pd
import numpy as np

# import matplotlib.pyplot as plt
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
    rbf_svm = svm.SVC(kernel='rbf')
    rbf_svm.fit(X_train, y_train)
    # mean accuracy on the given test data and labels
    score = rbf_svm.score(X_test, y_test) 
    print("mean acc =",score)


def main():
    simple_SVM()


if __name__ == "__main__": 
	main()