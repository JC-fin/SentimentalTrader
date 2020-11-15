import yfinance as yf
import pandas as pd 
import numpy as np
from TwelveDataWrapper import TwelveDataWrapper as tdw

FILE_NAME = '../data/msft_1day.csv'

class Trainer:
    def __init__(self, ticker):
        self.input_train = []
        self.output_train = []
        self.input_test  = []
        self.output_test = []
        self.ticker = ticker
        self.raw_data = pd.read_csv(FILE_NAME).set_index('Date')
        self.raw_data['change'] = self.raw_data['Close'] - self.raw_data['Open']
        self.raw_data['Increase'] = self.raw_data['Close'] > self.raw_data['Open']
        self.raw_data['Decrease'] = self.raw_data['Close'] < self.raw_data['Open']
        self.raw_data = self.raw_data.drop(['Open', 'Close', 'Dividends', 'Stock Splits'], axis=1)
        print(self.raw_data)
        trainTestSplit = int(len(self.raw_data) * 0.80)
        
        self.train_data = self.raw_data.iloc[:trainTestSplit]
        self.test_data = self.raw_data.iloc[trainTestSplit:]
        self.mean = self.train_data.mean(axis=0)
        self.std = self.train_data.std(axis=0)
        self.train_data.loc[:, 'Volume'] = ((self.train_data['Volume'] - self.mean.loc['Volume']) / self.std.loc['Volume']) 
        self.train_data.loc[:, 'change'] = ((self.train_data['change'] - self.mean.loc['change']) / self.std.loc['change']) 
        self.test_data.loc[:, 'Volume'] = ((self.test_data['Volume'] - self.mean.loc['Volume']) / self.std.loc['Volume']) 
        self.test_data.loc[:, 'change'] = ((self.test_data['change'] - self.mean.loc['change']) / self.std.loc['change']) 


        

    def getDataFromAPI(self):
            wrapper = tdw()
            stock_p1 = wrapper.time_series([self.ticker], interval='15min', start_date='2019-09-16 09:30:00', end_date='2020-04-22 15:30:00').sort_values('datetime', ascending=True).set_index('datetime')

            stock_p2 = wrapper.time_series([self.ticker], interval='15min', start_date='2020-04-22 15:45:00', end_date='2020-10-29').sort_values('datetime', ascending=True).set_index('datetime')

            macd_p1 = wrapper.macd(self.ticker, interval='15min', start_date='2019-09-16 09:30:00', end_date='2020-04-22 15:30:00').sort_values('datetime', ascending=True).set_index('datetime')

            macd_p2 = wrapper.macd(self.ticker, interval='15min', start_date='2020-04-22 15:45:00', end_date='2020-10-29').sort_values('datetime', ascending=True).set_index('datetime')

            stock_conglomerated = stock_p1.append(stock_p2).drop(['high', 'low'], axis=1)
            #macd_conglomerated = macd_p1.append(macd_p2)
            #print(pd.merge(stock_conglomerated, macd_conglomerated, left_index=True, right_index=True))
            #print(stock_conglomerated)
            stock_conglomerated.to_csv('../data/stockdata_15min.csv')


    
    def generateTrainData(self, seq_len):
        for i in range((len(self.train_data)//seq_len)*seq_len - seq_len - 1):
            x = np.array(self.train_data.iloc[i: i + seq_len, [0, 1]])
            y = np.array([self.train_data.iloc[i + seq_len, [2, 3]]], np.float64)
            self.input_train.append(x)
            self.output_train.append(y)
        self.X_train = np.array(self.input_train).reshape((len(self.input_train), seq_len, 2))# / np.array([250, 250, 1000000])
        self.Y_train = np.array(self.output_train).astype(np.float32).reshape((len(self.output_train), 2))
        #X = np.asarray(X).astype(np.float32)

    def generateTestData(self, seq_len):
        for i in range((len(self.test_data)//seq_len)*seq_len - seq_len - 1):
            x = np.array(self.test_data.iloc[i: i + seq_len, [0, 1]])
            y = np.array([self.test_data.iloc[i + seq_len, [2, 3]]], np.float64)
            self.input_test.append(x)
            self.output_test.append(y)
        self.X_test = np.array(self.input_test).reshape((len(self.input_test), seq_len, 2))# / np.array([250, 250, 1000000])
        self.Y_test = np.array(self.output_test).astype(np.float32).reshape((len(self.output_test), 2))

if __name__ == "__main__":
    msft = Trainer('MSFT')
    msft.generateTrainData(60)
    msft.generateTestData(60)
    #print(msft.Y_train)
    #print(msft.X_train)
    print(msft.Y_test)
