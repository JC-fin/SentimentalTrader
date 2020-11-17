#import yfinance as yf
import pandas as pd 
import numpy as np
from TwelveDataWrapper import TwelveDataWrapper as tdw
from sklearn.preprocessing import MinMaxScaler

FILE_NAME = '../data/MSFT.csv'

class Trainer:
    def __init__(self, ticker):
        wrapper = tdw()
        self.input_train = []
        self.input_technicals = []
        self.output_train = []
        self.input_test  = []
        self.output_test = []
        self.y_normalizer = None
        self.ticker = ticker
        
        raw_data1 = wrapper.time_series([ticker], interval='1day', start_date='2000-01-01', end_date='2019-11-15').drop('datetime', axis=1)
        raw_data2 = wrapper.time_series([ticker], interval='1day', start_date='2019-11-15', end_date='2020-11-12').drop('datetime', axis=1)
        self.raw_data = raw_data1.append(raw_data2)
        
        #self.raw_data = pd.read_csv(FILE_NAME).drop(['Date', 'Adj Close'],axis=1)
        print(self.raw_data)
        self.raw_data = self.raw_data.values
        trainTestSplit = int(len(self.raw_data) * 0.80)
        self.normalizer = MinMaxScaler().fit(self.raw_data)
        self.normalized_data = self.normalizer.transform(self.raw_data)    
        #print(self.normalized_data)    
        #self.train_data = self.raw_data.iloc[:trainTestSplit]
        #self.test_data = self.raw_data.iloc[trainTestSplit:]


        """
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
    """

        

    def getDataFromAPI(self):
            wrapper = tdw()
            stock_p1 = wrapper.time_series([self.ticker], interval='1day', start_date='2019-09-16 09:30:00', end_date='2020-04-22 15:30:00').sort_values('datetime', ascending=True).set_index('datetime')

            stock_p2 = wrapper.time_series([self.ticker], interval='15min', start_date='2020-04-22 15:45:00', end_date='2020-10-29').sort_values('datetime', ascending=True).set_index('datetime')

            macd_p1 = wrapper.macd(self.ticker, interval='15min', start_date='2019-09-16 09:30:00', end_date='2020-04-22 15:30:00').sort_values('datetime', ascending=True).set_index('datetime')

            macd_p2 = wrapper.macd(self.ticker, interval='15min', start_date='2020-04-22 15:45:00', end_date='2020-10-29').sort_values('datetime', ascending=True).set_index('datetime')

            stock_conglomerated = stock_p1.append(stock_p2).drop(['high', 'low'], axis=1)
            #macd_conglomerated = macd_p1.append(macd_p2)
            #print(pd.merge(stock_conglomerated, macd_conglomerated, left_index=True, right_index=True))
            #print(stock_conglomerated)
            stock_conglomerated.to_csv('../data/stockdata_15min.csv')


    
    def generateTrainData(self, seq_len):
        '''
        data_normalised = self.normalized_data
        data = self.raw_data
        ohlcv_histories_normalised = np.array([data_normalised[i:i + seq_len].copy() for i in range(len(data_normalised) - seq_len)])
        next_day_open_values_normalised = np.array([data_normalised[:, 0][i + seq_len].copy() for i in range(len(data_normalised) - seq_len)])
        next_day_open_values_normalised = np.expand_dims(next_day_open_values_normalised, -1)

        next_day_open_values = np.array([data[:, 0][i + seq_len].copy() for i in range(len(data) - seq_len)])
        next_day_open_values = np.expand_dims(next_day_open_values, -1)

        y_normaliser = MinMaxScaler()
        y_normaliser.fit(next_day_open_values)

        technical_indicators = []
        for his in ohlcv_histories_normalised:
            # note since we are using his[3] we are taking the SMA of the closing price
            sma = np.mean(his[:, 3])
            #macd = calc_ema(his, 12) - calc_ema(his, 26)
            technical_indicators.append(np.array([sma]))
            # technical_indicators.append(np.array([sma,macd,]))

        technical_indicators = np.array(technical_indicators)

        tech_ind_scaler = MinMaxScaler()
        technical_indicators_normalised = tech_ind_scaler.fit_transform(technical_indicators)
        self.X_train = ohlcv_histories_normalised
        self.X_technicals = technical_indicators_normalised
        self.Y_train = next_day_open_values_normalised
        self.y_normaliser = y_normaliser
        '''
        next_close = []
        for i in range((len(self.normalized_data)//seq_len)*seq_len - seq_len - 1):
            x = np.array(self.normalized_data[i: i + seq_len])
            sma = np.mean(x[:, 3])
            y = np.array([self.raw_data[i + seq_len, 0]])
            next_close.append(np.array(self.raw_data[i + seq_len, 0]))
            self.input_train.append(x)
            self.output_train.append(y)
            self.input_technicals.append(sma)
        self.X_train = np.array(self.input_train).reshape((len(self.input_train), seq_len, 5))
        self.X_technicals = np.array(self.input_technicals).reshape((len(self.input_technicals), 1))
        self.Y_train = np.array(self.output_train).reshape((len(self.output_train), 1))
        y_normalizer = np.expand_dims(next_close, -1)
        self.y_normalizer = MinMaxScaler()
        self.y_normalizer.fit(y_normalizer)
        self.Y_train = self.y_normalizer.transform(self.Y_train)
        print(self.y_normalizer.inverse_transform(self.Y_train))

        #X = np.asarray(X).astype(np.float32)

    def generateTestData(self, seq_len):
        for i in range((len(self.test_data)//seq_len)*seq_len - seq_len - 1):
            x = np.array(self.test_data.iloc[i: i + seq_len, [0, 1]])
            y = np.array([self.test_data.iloc[i + seq_len, [2, 3]]], np.float64)
            self.input_test.append(x)
            self.output_test.append(y)
        self.X_test = np.array(self.input_test).reshape((len(self.input_test), seq_len, 2))# / np.array([250, 250, 1000000])
        self.Y_test = np.array(self.output_test).astype(np.float32).reshape((len(self.output_test), 2))

def calc_ema(values, time_period):
        # https://www.investopedia.com/ask/answers/122314/what-exponential-moving-average-ema-formula-and-how-ema-calculated.asp
        sma = np.mean(values[:, 3])
        ema_values = [sma]
        k = 2 / (1 + time_period)
        for i in range(len(values) - time_period, len(values)):
            close = values[i][3]
            ema_values.append(close * k + ema_values[-1] * (1 - k))
        return ema_values[-1]

if __name__ == "__main__":
    msft = Trainer('MSFT')
    msft.generateTrainData(1)
    #msft.generateTestData(60)
    #print(msft.Y_train)
    #print(msft.X_train)

