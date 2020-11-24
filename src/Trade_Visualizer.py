import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
from TwelveDataWrapper import TwelveDataWrapper as tdw
from LSTMv2 import LSTMv2
from datetime import datetime
from pandas.tseries.offsets import BDay
from pandas_datareader import data

class Trade_Visualizer:
    def __init__(self):
        self.trade_hist_path = '../data/trade_history.json'
        self.visuals_path = '../visuals/'
        try:
            # trade history is dictionary: {ticker : (date, trade_price, pred_price, buy/sell/none)}
            self.trade_history = self.load_trade_history()
        except FileNotFoundError:
            self.trade_history = {}
        
    # Given a ticker, date, and either 'buy', 'sell', or 'np.nan' indicating
    # buy/sell, create a record containing the info in the class dict
    def add_trade(self, ticker, date, buy):
        date = str(date.date())
        ticker = ticker.lower()
        if ticker in self.trade_history:
            date_index = self.has_date(ticker, date)
            if date_index == -1:
                self.trade_history[ticker].append([date, np.nan, np.nan, buy])
            else:
                self.trade_history[ticker][date_index][3] = buy
        else:
            self.trade_history[ticker] = []
            self.trade_history[ticker].append([date, np.nan, np.nan, buy])
        self.save_trade_history()
        self.plot_buy_sell()

    # Given a ticker, date, stock price, and boolean indicating
    # buy/sell, create a record containing the info in the class dict
    def add_pred(self, ticker, date, price):
        date = str(date.date())
        ticker = ticker.lower()
        if ticker in self.trade_history:
            date_index = self.has_date(ticker, date)
            if date_index == -1:
                self.trade_history[ticker].append([date, np.nan, price, np.nan])
            else:
                self.trade_history[ticker][date_index][2] = price
        else:
            self.trade_history[ticker] = []
            self.trade_history[ticker].append([date, np.nan, price, np.nan])

    # Given a ticker, date, stock price, create a record containing the
    # info in the class dict
    def add_actual(self, ticker, date, price):
        date = str(date.date())
        ticker = ticker.lower()
        if ticker in self.trade_history:
            date_index = self.has_date(ticker, date)
            if date_index == -1:
                self.trade_history[ticker].append([date, price, np.nan, np.nan])
            else:
                self.trade_history[ticker][date_index][1] = price
        else:
            self.trade_history[ticker] = []
            self.trade_history[ticker].append([date, price, np.nan, np.nan])
        
    # Plots the quantitave predicted value vs the actual stock values and
    # saves them to the visualization directory with tha name <ticker>_predictions
    def plot_pred_vs_act(self):
        for key in self.trade_history.keys():
            df = pd.DataFrame(self.trade_history[key], columns=['date', 'act', 'pred', 'buy'])
            df['date'] = df['date'].apply((lambda x : datetime.strptime(x, '%Y-%m-%d')))
            df = df.set_index('date')
            df.sort_index(inplace=True)
            pred_vs_act = df[['act', 'pred']]
            pred_vs_act = pred_vs_act.dropna()
            fig, ax = plt.subplots()
            ax.plot(pred_vs_act.index, pred_vs_act['pred'].values, label='Predicted Price')
            ax.plot(pred_vs_act.index, pred_vs_act['act'].values, label='Actual Price')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.set_title(key + " Prediction History")
            ax.legend()
            plt.xticks(rotation='vertical')
            plt.subplots_adjust(left=0.1, bottom=0.2, right=0.9, top=0.9)
            plt.savefig(self.visuals_path + key + '_prediction.png')
            plt.close()
    
    # Creates plots for each company and saves them to the visualization 
    # directory with the name <ticker>_buysell
    def plot_buy_sell(self):
        for key in self.trade_history.keys():
            df = pd.DataFrame(self.trade_history[key], columns=['date', 'act', 'pred', 'buy'])
            if len(df) == 0:
                continue
            df['date'] = df['date'].apply((lambda x : datetime.strptime(x, '%Y-%m-%d')))
            df = df.set_index('date')
            df.sort_index(inplace=True)
            min_buy_df = df.loc[np.vectorize(pd.notna)(df['buy'])].sort_index()
            if len(min_buy_df) > 0:
                min_buy_index = min_buy_df.iloc[0].name
                df = df.loc[min_buy_index:]
            else:
                continue
            if len(df) == 0:
                continue
            act = df['act']
            act = act.dropna()
            buy_points = df.loc[df['buy'] == 'buy']['act']
            sell_points = df.loc[df['buy'] == 'sell']['act']
            buy_points = buy_points.dropna()
            sell_points = sell_points.dropna()
            fig, ax = plt.subplots()
            ax.plot(act.index, act.values, label='Actual Price')
            ax.scatter(buy_points.index, buy_points.values, label='Buy', c='g')
            ax.scatter(sell_points.index, sell_points.values, label='Sell', c='r')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.set_title(key + " Buy Sell History")
            ax.legend()
            plt.xticks(rotation='vertical')
            plt.subplots_adjust(left=0.1, bottom=0.2, right=0.9, top=0.9)
            plt.savefig(self.visuals_path + key + '_buysell.png')
            plt.close()
            
    # Reads the json file located at the given path and returns
    # a dictionary with company key as the ticker and a list of
    # tuples of (date, price, boolean true=buy false=sell) as values
    def load_trade_history(self):
        with open(self.trade_hist_path, 'r') as read_file:
            data = json.load(read_file)
        return data

    # Saves trade_history dictionary as json file in the given path
    def save_trade_history(self):
        with open(self.trade_hist_path, 'w') as write_file:
            json.dump(self.trade_history, write_file)
    
    # Return index of the given date  if the given ticker and given date already
    # exist in the trade history dict. If they do not, return -1
    def has_date(self, ticker, date):
        date = str(date)
        if not ticker in self.trade_history:
            return -1 
        for i in range(len(self.trade_history[ticker])):
            if self.trade_history[ticker][i][0] == date:
                return i
        return -1
   
    # Loads missing closing prices and predictions from the last stored closing price and
    # prediction to the most current closing price and predictions. Takes a dictionary 
    # containing trained LSTM models for tickers
    def update(self):
        for key in self.trade_history.keys():
            df = pd.DataFrame(self.trade_history[key], columns=['date', 'act', 'pred', 'buy'])
            df['date'] = df['date'].apply((lambda x : datetime.strptime(x, '%Y-%m-%d')))
            first_pred_miss = df.loc[df['pred'].isna()]['date'].max()
            first_act_miss = df.loc[df['act'].isna()]['date'].max()
            if (pd.isna(first_pred_miss) and pd.isna(first_act_miss)):
                #this is an arbitrary time to start tracing the history
                start_date = datetime(2020, 1, 2)
            else:
                start_date = min(first_pred_miss - BDay(1), first_act_miss)
            dates_needed = pd.date_range(str(start_date), str(datetime.today().date()), freq='B')
            wrapper = tdw()
            stock_info1 = wrapper.time_series([key], interval='1day', start_date='2000-01-01', end_date='2019-11-15')
            stock_info2 = wrapper.time_series([key], interval='1day', start_date='2019-11-11', end_date=str(datetime.today().date()))
            stock_info = stock_info1.append(stock_info2).reset_index()
            print("\n\nSTOCK INFO: ", stock_info)
            stock_info.set_index('datetime')
            # this variable keeps track of how many days old the model is for the current prediction
            # for big catch ups, ill just train the model every thirty days
            days_since_train = 30 
            for date in dates_needed:
                try:
                    raw_data = stock_info.iloc[:stock_info.loc[stock_info['datetime'] == str(date.date())].index.values[0] + 1]
                except IndexError:
                    continue
                print("\n\nRAW DATA: ", raw_data)
                raw_data = raw_data.drop(['datetime', 'index'], axis=1)
                if days_since_train == 30:
                    try:
                        cur_model = LSTMv2(key, raw_data=raw_data)
                        cur_model.trainModel()
                    except ValueError:
                        continue
                    days_since_train = 0
                prediction = cur_model.predictDay(raw_data.values[-1 * cur_model.histPoints:])[0][0]
                cur_price = stock_info.loc[stock_info['datetime'] == str(date.date())]['close'].values[0]
                self.add_pred(key, date + BDay(1), cur_price * prediction + cur_price)
                self.add_actual(key, date, cur_price)
                print('\n\nCURDATE: ' , date)
                print('TKR: ', key)
                print('Pchange: ', prediction)
                print('Cur_price: ', cur_price)
                print('prediction: ', cur_price * prediction + cur_price)
                days_since_train += 1
                
    # Loads todays closing prices and predictions given dictionary
    # containing trained LSTM models for tickers
    def update_today(self, LSTMs):
        for key in LSTMs.keys():
            prediction = LSTMs[key].predictNextDay()[0][0]
            print("COMPANY: ", key)
            print("DATA: ", LSTMs[key].trainer.ohlcv)
            cur_price = LSTMs[key].trainer.ohlcv[len(LSTMs[key].trainer.ohlcv) - 1, 3]
            self.add_pred(key, datetime.today() + BDay(1), cur_price * prediction + cur_price)
            self.add_actual(key, datetime.today(), cur_price)
        self.save_trade_history()

    # Writes over existing trade_history.json file with a json file containing all the desired tickers
    # with empty arrays as values. If tickers is none, new tickers are the same as the previous ones
    def reset_history(self, tickers=None):
        if tickers is None:
            tickers = self.trade_history.keys()
        self.trade_history = {}
        for ticker in tickers:
            self.trade_history[ticker.lower()] = []
        self.save_trade_history()

    # Given a ticker, returns the last non-nan prediction entry for that ticker
    def last_pred_date(self, ticker):
        th = pd.DataFrame(self.trade_history[ticker.lower()], columns=['date', 'act', 'pred', 'buy'])
        last_day = th.loc[th['pred'].notna()]['date'].max()
        return datetime.strptime(last_day, '%Y-%m-%d')

    # Given a ticker, returns the last non-nan prediction entry for that ticker
    def last_act_date(self, ticker):
        th = pd.DataFrame(self.trade_history[ticker.lower()], columns=['date', 'act', 'pred', 'buy'])
        last_day = th.loc[th['act'].notna()]['date'].max()
        return datetime.strptime(last_day, '%Y-%m-%d')
        







    





