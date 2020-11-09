import yfinance as yf
import pandas as pd
import numpy as np
from TwelveDataWrapper import TwelveDataWrapper as tdw

class Trainer:
    def __init__(self, ticker):
        self.input_train = []
        self.output_train = []
        self.input_test  = []
        self.output_test = []
        self.ticker = yf.Ticker(ticker)
        wrapper = tdw()
        self.stock_data = wrapper.time_series([ticker], interval='1day', start_date='2010-01-01', end_date='2019-10-12').sort_values('datetime', ascending=True).set_index('datetime')
        self.stock_data['change'] = (self.stock_data['close'] - self.stock_data['open']) / (self.stock_data['open'])
        low = np.percentile(np.array(self.stock_data['change']), 33.3333)
        high = np.percentile(np.array(self.stock_data['change']), 66.6666)
        print(low)
        print(high)
        self.stock_data.loc[self.stock_data['change'] > high, 'change'] = 1
        self.stock_data.loc[np.vectorize((lambda x : x <= high and x >= low))(self.stock_data['change']), 'change'] = 0
        self.stock_data.loc[self.stock_data['change'] < low, 'change'] = -1
        self.macd_data = wrapper.macd(ticker, interval='1day', start_date='2010-01-01', end_date='2019-10-12').sort_values('datetime', ascending=True).set_index('datetime')
        self.raw_data = pd.merge(self.stock_data, self.macd_data, right_index=True,
        left_index=True)
        #print(self.raw_data)
        #print("Data cols: ",  self.raw_data.columns)
        #print(self.raw_data)
        #self.raw_data = self.ticker.history(start="2013-01-01", end="2020-10-12").drop(['Dividends', 'Stock Splits'], axis=1)
        trainTestSplit = int(len(self.raw_data) * 0.80)
        self.raw_data = self.raw_data.dropna(axis=0)
        self.train_data = self.raw_data.iloc[:trainTestSplit]
        self.test_data = self.raw_data.iloc[trainTestSplit:]

    def generateTrainData(self, seq_len):
        for i in range((len(self.train_data)//seq_len)*seq_len - seq_len - 1):
            x = np.array((self.train_data.iloc[i: i + seq_len, [3, 4, 6, 7, 8]]).apply((lambda x : (x - x.min()) / (x.max() - x.min())), axis=0))
            y = (lambda x : np.array([int (x == -1), int(x == 0), int(x == 1)]))(self.train_data.iloc[i + seq_len, 5])
            self.input_train.append(x)
            self.output_train.append(y)
        self.X_train = np.array(self.input_train).reshape((len(self.input_train), seq_len, 5))
        self.Y_train = np.array(self.output_train)

    def generateTestData(self, seq_len):
        for i in range((len(self.test_data)//seq_len)*seq_len - seq_len - 1):
            x = np.array((self.test_data.iloc[i: i + seq_len, [3, 4, 6, 7, 8]]).apply((lambda x : (x - x.min()) / (x.max() - x.min())), axis=0))
            y = (lambda x : np.array([int (x == -1), int(x == 0), int(x == 1)]))(self.test_data.iloc[i + seq_len, 5])
            self.input_test.append(x)
            self.output_test.append(y)
        self.X_test = np.array(self.input_test).reshape((len(self.input_test), seq_len, 5))
        self.Y_test = np.array(self.output_test)

if __name__ == "__main__":
    msft = Trainer('MSFT')
    msft.generateTrainData(60)
    msft.generateTestData(60)
    print(msft.Y_train)
    print(msft.X_train)
