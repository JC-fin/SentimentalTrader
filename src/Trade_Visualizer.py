import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

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
                self.trade_history[ticker].append((date, np.nan, np.nan, buy))
            else:
                self.trade_history[ticker][date_index][3] = buy
        else:
            self.trade_history[ticker] = []
            self.trade_history[ticker].append([date, np.nan, np.nan, buy])

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
            plt.savefig(self.visuals_path + key + '_prediction.png')
    
    # Creates plots for each company and saves them to the visualization 
    # directory with the name <ticker>_buysell
    def plot_buy_sell(self):
        for key in self.trade_history.keys():
            df = pd.DataFrame(self.trade_history[key], columns=['date', 'act', 'pred', 'buy'])
            df['date'] = df['date'].apply((lambda x : datetime.strptime(x, '%Y-%m-%d')))
            df = df.set_index('date')
            df.sort_index(inplace=True)
            min_buy_index = df.loc[np.vectorize(pd.isna)(df['buy'])].sort_index().iloc[0].name
            df = df.loc[min_buy_index:]
            act = df['act']
            act = act.dropna()
            buy_points = df.loc[df['buy'] == 'buy']['act']
            sell_points = df.loc[df['buy'] == 'sell']['act']
            fig, ax = plt.subplots()
            ax.plot(act.index, act.values, label='Actual Price')
            ax.scatter(buy_points.index, buy_points.values, label='Buy', c='g')
            ax.scatter(sell_points.index, sell_points.values, label='Sell', c='r')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.set_title(key + " Buy Sell History")
            ax.legend()
            plt.savefig(self.visuals_path + key + '_buysell.png')
            
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
    # prediction to the most current closing price and predictions
    #def update(self, tickers):
        

    # Load today's prices closing prices into dictionary


    # Load tomorrow's predictions into dictionary






    





