import alpaca_trade_api as api
from TwelveDataWrapper import TwelveDataWrapper as tdw
from Trade_Vizualizer import Trade_Vizualizer as tv
from datetime import date, timedelta
from LSTMv2 import LSTMv2

KEY_ID = "PKM8TR5SZ1SR1BZ39P6N"
KEY_SECRET = "v3sXwbjf3b4nrmnJEbc2Sso4AuBBa3BYrM8RiKbN"
URL = "https://paper-api.alpaca.markets"

class TradingBot:
    def __init__(self, tickers) :
        self.alpaca = api.REST(KEY_ID, KEY_SECRET, URL, 'v2')
        self.LSTMs = {}
        self.buy_sell_viz = tv()
        self.tickers = tickers
        for ticker in tickers:
            self.LSTMs[ticker] = LSTMv2(ticker, str(date.now()))
            self.LSTMs[ticker].trainModel()
        self.tdw = tdw()
    
    def buy(self, ticker, qty):
        self.alpaca.submit_order(ticker, qty, "buy", "market", "day")
        self.buy_sell_viz.add_trade(ticker, date.today(), 'buy')

    def sell(self, ticker, qty):
        self.alpaca.submit_order(ticker, qty, "sell", "market", "day")
        self.buy_sell_viz.add_trade(ticker, date.today(), 'sell')
    
    def predictMovement(self, ticker, model):
        prediction = model.predictNextDay()
        return prediction
    
    def analyzeStocks(self):
        predictions = {}
        for ticker in self.tickers:
            model = self.LSTMs.get(ticker)
            predictions[ticker] = self.predictMovement(ticker, model)
        return predictions
