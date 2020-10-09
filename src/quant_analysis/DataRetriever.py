import pandas as pd
import numpy as np
from TwelveDataWrapper import TwelveDataWrapper

class DataRetriever:
    def __init__(self, tickers):
        self.time_series = {}
        self.twelveDataWrapper = TwelveDataWrapper()
        for ticker in tickers:
            self.time_series[ticker] = self.twelveDataWrapper.time_series([ticker])
    
    def printTickerInfo(self, ticker):
        print(self.time_series.get(ticker))

test1 = DataRetriever(["MSFT", "AAPL", "NVDA", "NKLA"])
test1.printTickerInfo("NKLA")