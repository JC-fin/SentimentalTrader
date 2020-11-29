import pandas as pd 
import numpy as np
from TwelveDataWrapper import TwelveDataWrapper as tdw
from sklearn.preprocessing import MinMaxScaler

FILE_NAME = '../data/MSFT.csv'

class Trainer:
    def __init__(self, ticker, raw_data=None):
        wrapper = tdw()
        self.input_train = []
        self.input_technicals = []
        self.output_train = []
        self.input_test  = []
        self.output_test = []
        self.y_normalizer = None
        self.ticker = ticker
        if raw_data is None:
            raw_data1 = wrapper.time_series([ticker], interval='1day', start_date='2000-01-01', end_date='2019-11-15').drop('datetime', axis=1)
            raw_data2 = wrapper.time_series([ticker], interval='1day', start_date='2019-11-15').drop('datetime', axis=1)
            self.raw_data = raw_data1.append(raw_data2)
        else:
            self.raw_data = raw_data
        self.ohlcv = self.raw_data.values
        self.raw_data['change'] = (self.raw_data['close'] - self.raw_data['open']) / self.raw_data['open']
        self.raw_data = self.raw_data.drop(['open', 'close', 'high', 'low'], axis=1)
        self.raw_data = self.raw_data.values
        trainTestSplit = int(len(self.raw_data) * 0.80)
        self.normalizer = MinMaxScaler().fit(self.raw_data[:, [0]])
        self.normalized_data = np.hstack((self.normalizer.transform(self.raw_data[:, [0]]), self.raw_data[:, [1]]))    
        print(self.normalized_data)    

    def getDataFromAPI(self):
        wrapper = tdw()
        stock_p1 = wrapper.time_series([self.ticker], interval='1day', start_date='2019-09-16 09:30:00', end_date='2020-04-22 15:30:00').sort_values('datetime', ascending=True).set_index('datetime')

        stock_p2 = wrapper.time_series([self.ticker], interval='15min', start_date='2020-04-22 15:45:00', end_date='2020-10-29').sort_values('datetime', ascending=True).set_index('datetime')

        macd_p1 = wrapper.macd(self.ticker, interval='15min', start_date='2019-09-16 09:30:00', end_date='2020-04-22 15:30:00').sort_values('datetime', ascending=True).set_index('datetime')

        macd_p2 = wrapper.macd(self.ticker, interval='15min', start_date='2020-04-22 15:45:00', end_date='2020-10-29').sort_values('datetime', ascending=True).set_index('datetime')

        stock_conglomerated = stock_p1.append(stock_p2).drop(['high', 'low'], axis=1)
        stock_conglomerated.to_csv('../data/stockdata_15min.csv')


    
    def generateTrainData(self, seq_len):
        macd = []
        for i in range((len(self.normalized_data)//seq_len)*seq_len - seq_len - 1):
            x = np.array(self.normalized_data[i: i + seq_len])
            y = np.array([self.raw_data[i + seq_len, 1]])
            if (i + seq_len >= 26):
                ema12 = self.expoMovingAvg(12, i + seq_len)
                ema26 = self.expoMovingAvg(26, i + seq_len)
            else:
                ema12 = 0
                ema26 = 0
            macd.append(ema26 - ema12)
            self.input_train.append(x)
            self.output_train.append(y)
        self.X_technicals = np.array(macd).reshape((len(macd), 1))
        self.X_train = np.array(self.input_train).reshape((len(self.input_train), seq_len, 2))
        self.Y_train = np.array(self.output_train).reshape((len(self.output_train), 1))

    def generateTestData(self, seq_len):
        for i in range((len(self.test_data)//seq_len)*seq_len - seq_len - 1):
            x = np.array(self.test_data.iloc[i: i + seq_len, [0, 1]])
            y = np.array([self.test_data.iloc[i + seq_len, [2, 3]]], np.float64)
            self.input_test.append(x)
            self.output_test.append(y)
        self.X_test = np.array(self.input_test).reshape((len(self.input_test), seq_len, 2))
        self.Y_test = np.array(self.output_test).astype(np.float32).reshape((len(self.output_test), 2))

    def expoMovingAvg(self, period, end):
        # ohcv -> [open, high, low, close]
        if (end < period) :
            return 0
        ohcv = self.ohlcv[(end - period) : end]
        sma = np.mean(ohcv[:, 3])
        ema = [sma]
        mul = 2 / (1 + period)
        for i in range(len(ohcv) - period, len(ohcv)):
            close = ohcv[i][3]
            ema.append(close * mul + ema[-1] * (1 - mul))
        return ema[-1]
if __name__ == "__main__":
    msft = Trainer('MSFT')
    msft.generateTrainData(20)
    print(msft.X_technicals)
    #msft.generateTestData(60)
    #print(msft.Y_train)
    #print(msft.X_train)

