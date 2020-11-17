from sentimentTrainer import SentimentTrainer
from dataLoader import DataLoader
from sentimentPredictor import SentimentPredictor
import sys
sys.path.append("../webscraping/")
from scrapeTitles import TitleScraper
import pandas as pd
import numpy as np

companies = {'nikola': 'NKLA', 'microsoft': 'MSFT', 'apple': 'AAPL',
                           'netflix': 'NFLX', 'workday': 'WDAY', 'nvidia': 'NVDA', 'norton': 'NLOK', 'xerox': 'XRX',
                           'hp': 'HPQ', 'micro': 'AMD', 'advanced': 'AMD', 'moderna': 'MRNA', 'peloton': 'PTON', 'home': 'HD', 'depot': 'HD'}

df = pd.DataFrame(columns=['Date', 'Ticker', 'Headline'])

for key in companies:
    ts = TitleScraper(companies[key], key, "11/04/2020", "11/05/2020", 20)
    ts.main()
    frame = pd.DataFrame({'Date': pd.Series(["11/05/2020"]).repeat(len(ts.getTitleList())),
    'Ticker': pd.Series(companies[key]).repeat(len(ts.getTitleList())),
    'Headline': ts.getTitleList()})
    df = df.append(frame, ignore_index=True)

dl = DataLoader()
dl.load_vocab('../../data/finData/pos', '../../data/finData/neg')

# st = SentimentTrainer(dl.vocab)
# st.train_model('../../data/finData/pos', '../../data/finData/neg')

# df = pd.DataFrame({'Date':['10-21-2020'], 'Ticker': ['MCFE'], 'Headline':"McAfee's IPO Raises $740 Million in Return to Public Market"}) # clear positive
# df = df.append(pd.DataFrame({'Date':['10-21-2020'], 'Ticker': ['TSLA'], 'Headline':"Tesla Beats on Profit, Reaffirms Goal of 500,000 Deliveries"})) #clear positive
# df = df.append(pd.DataFrame({'Date':['10-21-2020'], 'Ticker': ['QUIB'], 'Headline':"Quibi Will Close Down in One of Hollywood's Biggest Flops"})) #clear negative
# df = df.append(pd.DataFrame({'Date':['10-21-2020'], 'Ticker': ['EQUI'], 'Headline':"Equinox Taps Kirkland & Ellis, Centerview for Debt Advice"})) #iffy negative
# df = df.append(pd.DataFrame({'Date':['10-21-2020'], 'Ticker': ['XOM'], 'Headline':"Exxon CEO Plans Layoffs, Underscores Faith in Fossil Fuels"})) #clear negative
# df = df.append(pd.DataFrame({'Date':['10-21-2020'], 'Ticker': ['GM'], 'Headline':"GM sells out first year of electric Hummer production"})) #clear positive
# df = df.append(pd.DataFrame({'Date':['10-21-2020'], 'Ticker': ['ALI'], 'Headline':"Align Tech Soars Past $400 as Sales 'Knocked It Out of the Park'"})) #clear positive
# df = df.append(pd.DataFrame({'Date':['10-21-2020'], 'Ticker': ['PHAR'], 'Headline':"Purdue Pharma to Plead Guilty, Pay $8.3 Billion Over Opiods"})) #clear negative

sp = SentimentPredictor(dl.vocab)
print(sp.predict_sentiment(df))

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

print(df.groupby('Ticker')['Prediction'].apply(mean))
print(df.groupby('Ticker')['Prediction'].apply(median))