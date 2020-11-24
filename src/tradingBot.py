import alpaca_trade_api as api
from TwelveDataWrapper import TwelveDataWrapper as tdw
from Trade_Visualizer import Trade_Visualizer as tv
from datetime import datetime, timedelta
from LSTMv2 import LSTMv2

KEY_ID = "PKUSO4DJQOO2NNX17NKG"
KEY_SECRET = "oS7d1pGVruPH4At5KzZ4ibTagik0Qy0o6ciqsZf5"
URL = "https://paper-api.alpaca.markets"

class TradingBot:
    def __init__(self, tickers) :
        self.alpaca = api.REST(KEY_ID, KEY_SECRET, URL, 'v2')
        self.LSTMs = {}
        self.buy_sell_viz = tv()
        for ticker in tickers:
            try:
                self.LSTMs[ticker] = LSTMv2(ticker)
                self.LSTMs[ticker].trainModel()
            except KeyError:
                self.LSTMs.pop(ticker, None)
                continue
        self.tickers = self.LSTMs.keys()
        self.buy_sell_viz.update_today(self.LSTMs)
        self.tdw = tdw()
    
    def buy(self, ticker, qty):
        self.alpaca.submit_order(ticker, qty, "buy", "market", "day")
        self.buy_sell_viz.add_trade(ticker, self.buy_sell_viz.last_pred_date(ticker), 'buy')
        
    def sell(self, ticker, qty):
        self.alpaca.submit_order(ticker, qty, "sell", "market", "day")
        self.buy_sell_viz.add_trade(ticker, self.buy_sell_viz.last_pred_date(ticker), 'sell')
    
    def predictMovement(self, ticker, model):
        prediction = model.predictNextDay()
        return prediction
    
    def analyzeStocks(self):
        predictions = {}
        for ticker in self.tickers:
            model = self.LSTMs.get(ticker)
            predictions[ticker] = self.predictMovement(ticker, model)
        return predictions
