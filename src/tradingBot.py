import alpaca_trade_api as api

KEY_ID = "PKM8TR5SZ1SR1BZ39P6N"
KEY_SECRET = "v3sXwbjf3b4nrmnJEbc2Sso4AuBBa3BYrM8RiKbN"
URL = "https://paper-api.alpaca.markets"

class TradingBot:
    def __init__(self) :
        self.alpaca = api.REST(KEY_ID, KEY_SECRET, URL, 'v2')
        #self.LSTM = LSTM('SPY')
    
    def buy(self, ticker, qty):
        self.alpaca.submit_order(ticker, qty, "buy", "market", "day")

    def sell(self, ticker, qty):
        self.alpaca.submit_order(ticker, qty, "sell", "market", "day")
test = TradingBot()
test.sell('MSFT', 1)

