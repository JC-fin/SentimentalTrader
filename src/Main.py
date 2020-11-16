from tradingBot import TradingBot
from neuralNet.sentimentTrainer import SentimentTrainer
from neuralNet.dataLoader import DataLoader
from neuralNet.sentimentPredictor import SentimentPredictor
from webscraping.scrapeTitles import titleScraper
from datetime import date
from datetime import timedelta
import pandas as pd
import numpy as np

def median(arrs):
    arrs.dropna(inplace=True)
    arrs.sort_values(inplace=True)
    size = len(arrs)
    return arrs.iloc[size // 2]

def main():
    train_model = False
    training_data_dir = '../data/finData/'
    # tickers = {
    #     'NKLA' : 'Nikola',
    #     'MSFT' : 'Microsoft',
    #     'AAPL' : 'Apple', 
    #     'NFLX' : 'Netflix',
    #     'WDAY' : 'Workday',
    #     'NVDA' : "Nvidia",
    #     'NLOK' : 'Norton',
    #     'XRX'  : 'Xerox',
    #     'HPQ'  : 'HP',
    #     'AMD'  : 'AMD',
    #     'MRNA' : 'Moderna',
    #     'PTON' : 'Peloton',
    #     'HD'   : 'Home Depot'
    # }

    tickers = {'NFLX': 'Netflix'}

    df = pd.DataFrame(columns=['Date', 'Ticker', 'Headline'])

    for key in tickers:
        # get 20 titles for each ticker in tickers from the last 2 days
        ts = titleScraper(key, tickers[key], (date.today() - timedelta(days=2)).strftime('%m/%d/%Y'), date.today().strftime('%m/%d/%Y'), 50)
        ts.main()
        frame = pd.DataFrame({'Date': pd.Series([date.today().strftime('%m/%d/%Y')]).repeat(len(ts.getTitleList())),
        'Ticker': pd.Series(key).repeat(len(ts.getTitleList())),
        'Headline': ts.getTitleList()})
        df = df.append(frame, ignore_index=True)

    
    
    print(df)
    dl = DataLoader()
    dl.load_vocab(training_data_dir + 'pos', training_data_dir + 'neg')

    if train_model:
        st = SentimentTrainer(dl.vocab)
        st.train_model(training_data_dir + 'pos', training_data_dir + 'neg')

    sp = SentimentPredictor(dl.vocab)
    sp.predict_sentiment(df)

    medianPred = df.groupby('Ticker')['Prediction'].apply(median).rename('median')
    
    trader = TradingBot(tickers.keys())
    predictions = trader.analyzeStocks()
    for ticker in predictions.keys():
        result = (predictions[ticker] + medianPred[ticker]) / 2
        if result > 0.5:
            trader.buy(ticker, int(abs(0.5 - result) * 20))
        else:
            trader.sell(ticker, int(abs(0.5 - result) * 20))

if __name__ == "__main__":
    main()