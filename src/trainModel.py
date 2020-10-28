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
        print(self.stock_data['change'])
        self.macd_data = wrapper.macd(ticker, interval='1day', start_date='2010-01-01', end_date='2019-10-12').sort_values('datetime', ascending=True).set_index('datetime')
        self.raw_data = pd.merge(self.stock_data, self.macd_data, right_index=True,
        left_index=True)
        #print(self.raw_data)
        #print("Data cols: ",  self.raw_data.columns)
        #print(self.raw_data)
        #self.raw_data = self.ticker.history(start="2013-01-01", end="2020-10-12").drop(['Dividends', 'Stock Splits'], axis=1)
        trainTestSplit = int(len(self.raw_data) * 0.80)
        
        self.train_data = self.raw_data.iloc[:trainTestSplit]
        self.test_data = self.raw_data.iloc[trainTestSplit:]
        
    
    def generateTrainData(self, seq_len):
        for i in range((len(self.train_data)//seq_len)*seq_len - seq_len - 1):
            x = np.array(self.raw_data.iloc[i: i + seq_len, [3, 4, 5, 6, 7]])
            y = np.array([self.raw_data.iloc[i + seq_len + 1, 3]], np.float64)
            self.input_train.append(x)
            self.output_train.append(y)
        self.X_train = np.array(self.input_train).reshape((len(self.input_train), seq_len, 5)) / np.array([250, 10000000, 1, 1, 1])
        self.Y_train = np.array(self.output_train) / 250

    def generateTestData(self, seq_len):
        for i in range((len(self.test_data)//seq_len)*seq_len - seq_len - 1):
            x = np.array(self.test_data.iloc[i: i + seq_len, [3, 4, 5, 6, 7]])
            y = np.array([self.test_data.iloc[i + seq_len + 1, 3]], np.float64)
            self.input_test.append(x)
            self.output_test.append(y)
        self.X_test = np.array(self.input_test).reshape((len(self.input_test), seq_len, 5)) / np.array([250, 10000000, 1,  1, 1])
        self.Y_test = np.array(self.output_test) / 250

if __name__ == "__main__":
    msft = Trainer('MSFT')
    msft.generateTrainData(60)
    msft.generateTestData(60)
    print(msft.Y_train)
    print(msft.X_train)