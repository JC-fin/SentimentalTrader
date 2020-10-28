import os
import requests
import pandas as pd
import numpy as np
from io import StringIO

TWELVE_DATA_KEY = "5772c82dd8ed4a65b6ac4719baad98db"

class TwelveDataWrapper:
    def __init__(self):
        self.key = TWELVE_DATA_KEY

    # Returns time-series data in a pandas dataframe
    def time_series(self, tickers, interval="15min", outputSize=None,
        start_date=None, end_date=None, _format='CSV'):
        symbols = ",".join(tickers)
        url = ("https://api.twelvedata.com/time_series?symbol=%s&interval=%s&apikey=%s&format=%s"
            % (symbols, interval, self.key, _format))

        if start_date is not None:
            url += "&start_date=%s" % (start_date)
        if end_date is not None:
            url += "&end_date=%s" % (end_date)
        if outputSize is not None:
            url += "&outputsize=%d" % outputSize
    
        response = requests.get(url)
        DATA = StringIO(str(response.text))
        return pd.read_csv(DATA, sep=";")

    def macd(self,symbol, interval="15min", outputSize=None,
        start_date=None, end_date=None, _format='CSV'):
        url = ("https://api.twelvedata.com/macd?symbol=%s&interval=%s&apikey=%s&format=%s" % (symbol, interval, self.key, _format))
        
        if start_date is not None:
            url += "&start_date=%s" % (start_date)
        if end_date is not None:
            url += "&end_date=%s" % (end_date)
        if outputSize is not None:
            url += "&outputsize=%d" % outputSize
    
        response = requests.get(url)
        DATA = StringIO(str(response.text))
        return pd.read_csv(DATA, sep=";")



def newWrapper():
    return Wrapper()