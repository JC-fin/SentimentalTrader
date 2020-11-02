import yfinance as yf
import pandas as pd 
import numpy as np
from TwelveDataWrapper import TwelveDataWrapper as tdw

FILE_NAME = '../data/stockdata_15min.csv'

class Trainer:
    def __init__(self, ticker):
        self.input_train = []
        self.output_train = []
        self.input_test  = []
        self.output_test = []
        self.ticker = ticker
        self.raw_data = pd.read_csv(FILE_NAME).set_index('datetime')
        trainTestSplit = int(len(self.raw_data) * 0.80)
        
        self.train_data = self.raw_data.iloc[:trainTestSplit]
        self.test_data = self.raw_data.iloc[trainTestSplit:]

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
            x = np.array(self.raw_data.iloc[i: i + seq_len])
            y = np.array([self.raw_data.iloc[i + seq_len + 1, 1]], np.float64)
            self.input_train.append(x)
            self.output_train.append(y)
        self.X_train = np.array(self.input_train).reshape((len(self.input_train), seq_len, 3)) / np.array([250, 250, 1000000])
        self.Y_train = np.array(self.output_train) / 250

    def generateTestData(self, seq_len):
        for i in range((len(self.test_data)//seq_len)*seq_len - seq_len - 1):
            x = np.array(self.test_data.iloc[i: i + seq_len])
            y = np.array([self.test_data.iloc[i + seq_len + 1, 1]], np.float64)
            self.input_test.append(x)
            self.output_test.append(y)
        self.X_test = np.array(self.input_test).reshape((len(self.input_test), seq_len, 3)) / np.array([250, 250, 1000000])
        self.Y_test = np.array(self.output_test) / 250

if __name__ == "__main__":
    msft = Trainer('MSFT')
    msft.generateTrainData(60)
    msft.generateTestData(60)
    #print(msft.Y_train)
    #print(msft.X_train)