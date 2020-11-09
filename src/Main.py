from tradingBot import tradingBot

def main():
    tickers = [
        'NKLA',
        'MSFT',
        'AAPL', 
        'NFLX',
        'WDAY',
        'NVDA',
        'NLOC',
        'XRX',
        'HPQ',
        'AMD',
        'MDNA',
        'PLT',
        'HD'
    ]
    trader = tradingBot(tickers)
    predictions = trader.analyzeStocks()
    for ticker in predictions.keys():
        if predictions[ticker] > 0:
            trader.buy(ticker)
        else:
            trader.sell(ticker)