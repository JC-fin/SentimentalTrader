from tradingBot import TradingBot
from neuralNet.sentimentTrainer import SentimentTrainer
from neuralNet.dataLoader import DataLoader
from neuralNet.sentimentPredictor import SentimentPredictor
from webscraping.scrapeTitles import titleScraper
import pandas as pd
import numpy as np

def mean(arrs):
        sum = 0
        total = 0
        for arr in arrs:
            if not np.isnan(arr):
                sum += arr
                total += 1
        return sum / total

def median(arrs):
    arrs.dropna(inplace=True)
    arrs.sort_values(inplace=True)
    size = len(arrs)
    return arrs.iloc[size // 2]

def main():
    train_model = False
    training_data_dir = '../data/finData/'
    tickers = {
        'NKLA' : 'Nikola',
        'MSFT' : 'Microsoft',
        'AAPL' : 'Apple', 
        'NFLX' : 'Netflix',
        'WDAY' : 'Workday',
        'NVDA' : "Nvidia",
        'NLOK' : 'Norton',
        'XRX'  : 'Xerox',
        'HPQ'  : 'HP',
        'AMD'  : 'AMD',
        'MRNA' : 'Moderna',
        'PTON' : 'Peloton',
        'HD'   : 'Home Depot'
    }

    df = pd.DataFrame(columns=['Date', 'Ticker', 'Headline'])

    for key in tickers:
        ts = titleScraper(key, tickers[key],  "11/04/2020", "11/05/2020", 20)
        ts.main()
        frame = pd.DataFrame({'Date': pd.Series(["11/05/2020"]).repeat(len(ts.getTitleList())),
        'Ticker': pd.Series(key).repeat(len(ts.getTitleList())),
        'Headline': ts.getTitleList()})
        df = df.append(frame, ignore_index=True)

    dl = DataLoader()
    dl.load_vocab(training_data_dir + 'pos', training_data_dir + 'neg')

    if train_model:
        st = SentimentTrainer(dl.vocab)
        st.train_model(training_data_dir + 'pos', training_data_dir + 'neg')

    sp = SentimentPredictor(dl.vocab)
    sp.predict_sentiment(df)

    meanPred = df.groupby('Ticker')['Prediction'].apply(mean).rename('mean')
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