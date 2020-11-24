import alpaca_trade_api as api
import numpy as numpy
import pandas as pd
from TwelveDataWrapper import TwelveDataWrapper as tdw
from datetime import date, timedelta
from LSTMv2 import LSTMv2
import time

KEY_ID = "PKUSO4DJQOO2NNX17NKG"
KEY_SECRET = "oS7d1pGVruPH4At5KzZ4ibTagik0Qy0o6ciqsZf5"
URL = "https://paper-api.alpaca.markets"

class TradingBot:
    def __init__(self, tickers) :
        self.alpaca = api.REST(KEY_ID, KEY_SECRET, URL, 'v2')
        self.loadPositions()
        self.LSTMs = {}
        self.tickers = tickers
        for ticker in tickers:
            self.LSTMs[ticker] = LSTMv2(ticker)
            self.LSTMs[ticker].trainModel()
            time.sleep(60)
        self.tdw = tdw()
    
    def buy(self, ticker, qty):
        self.alpaca.submit_order(ticker, qty, "buy", "market", "day")

    def sell(self, ticker, qty):
        if ticker in self.positions.index:
            toSell = min(self.positions.loc[ticker], qty)
            self.alpaca.submit_order(ticker, toSell, "sell", "market", "day")
    
    def predictMovement(self, ticker, model):
        prediction = model.predictNextDay()
        return prediction
    
    def analyzeStocks(self):
        predictions = {}
        for ticker in self.tickers:
            model = self.LSTMs.get(ticker)
            predictions[ticker] = self.predictMovement(ticker, model)
        return predictions

    def loadPositions(self):
        positions = self.alpaca.list_positions()
        tickers = []
        held = []
        for pos in positions:
            tickers.append(pos.symbol)
            held.append(int(pos.qty))
        self.positions = pd.Series(held, index=tickers)

if __name__ == '__main__':
    pac = TradingBot(['MSFT'])
    print(pac.getPositions())
