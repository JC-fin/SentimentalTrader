from sentimentTrainer import SentimentTrainer
from dataLoader import DataLoader
from sentimentPredictor import SentimentPredictor
import pandas as pd

dl = DataLoader()
dl.load_vocab('../../data/finData/pos', '../../data/finData/neg')

st = SentimentTrainer(dl.vocab)
st.train_model('../../data/finData/pos', '../../data/finData/neg')

df = pd.DataFrame({'Date':['10-21-2020'], 'Ticker': ['MCFE'], 'Headline':"McAfee's IPO Raises $740 Million in Return to Public Market"}) # clear positive
df = df.append(pd.DataFrame({'Date':['10-21-2020'], 'Ticker': ['TSLA'], 'Headline':"Tesla Beats on Profit, Reaffirms Goal of 500,000 Deliveries"})) #clear positive
df = df.append(pd.DataFrame({'Date':['10-21-2020'], 'Ticker': ['QUIB'], 'Headline':"Quibi Will Close Down in One of Hollywood's Biggest Flops"})) #clear negative
df = df.append(pd.DataFrame({'Date':['10-21-2020'], 'Ticker': ['EQUI'], 'Headline':"Equinox Taps Kirkland & Ellis, Centerview for Debt Advice"})) #iffy negative
df = df.append(pd.DataFrame({'Date':['10-21-2020'], 'Ticker': ['XOM'], 'Headline':"Exxon CEO Plans Layoffs, Underscores Faith in Fossil Fuels"})) #clear negative
df = df.append(pd.DataFrame({'Date':['10-21-2020'], 'Ticker': ['GM'], 'Headline':"GM sells out first year of electric Hummer production"})) #clear positive
df = df.append(pd.DataFrame({'Date':['10-21-2020'], 'Ticker': ['ALI'], 'Headline':"Align Tech Soars Past $400 as Sales 'Knocked It Out of the Park'"})) #clear positive
df = df.append(pd.DataFrame({'Date':['10-21-2020'], 'Ticker': ['PHAR'], 'Headline':"Purdue Pharma to Plead Guilty, Pay $8.3 Billion Over Opiods"})) #clear negative

sp = SentimentPredictor(dl.vocab)
print(sp.predict_sentiment(df))
