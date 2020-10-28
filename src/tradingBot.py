import alpaca_trade_api as api
from LSTM import LSTM

KEY_ID = "PKM8TR5SZ1SR1BZ39P6N"
KEY_SECRET = "v3sXwbjf3b4nrmnJEbc2Sso4AuBBa3BYrM8RiKbN"
URL = "https://paper-api.alpaca.markets"

class TradingBot:
    def __init__(self) :
        self.alpaca = api.REST(KEY_ID, KEY_SECRET, URL, 'v2')
        self.LSTM = LSTM('SPY')
