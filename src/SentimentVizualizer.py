
from IPython import get_ipython
from tradingBot import TradingBot
from neuralNet.sentimentTrainer import SentimentTrainer
from neuralNet.dataLoader import DataLoader
from neuralNet.sentimentPredictor import SentimentPredictor
from webscraping.scrapeTitles import TitleScraper
from datetime import date
from datetime import timedelta
import pandas as pd
import numpy as np
import os
from scipy import stats
import matplotlib.pyplot as plt
import mplcursors
import scipy.interpolate as interpolate


class SentimentVizualizer:

    def __init__(self, numHeadlines, days, tickerList):
        """
            Function: Vizualize company sentiment by creating instance and calling main()


            ex: numheadlines = 10, days = 30, tickerList = [{'HD' : 'HomeDepot}] 
            scrapes, predicts, and plots sentiment for HomeDepot 
            scraping 10 articles for each of the past 30 days

            numHeadlines --> number of headlines to scrape each day

            days --> number of days in the past to plot

            tickerList --> list of dicts, ticker : name
                    
        """
       
        self.train_model = False
        self.numHeadlines = numHeadlines
        self.daysToPredict = days
        self.training_data_dir = '../data/finData/'
        self.tickerList = tickerList


    def getTitleDF(self, tickers, start, end):
        """
        tickers --> dictionary ticker : company name

        start--> scrape start date 

        end --> scrape end date

        returns--> dataframe of headlines and the matching ticker columns = [Ticker, Headline]
        """

        df = pd.DataFrame(columns=['Date', 'Ticker', 'Headline'])

        for key in tickers:

            ts = TitleScraper(key, tickers[key], start, end, self.numHeadlines)

            ts.main()

            frame = pd.DataFrame({'Date': pd.Series([end]).repeat(len                                       (ts.getTitleList())),
                'Ticker': pd.Series(key).repeat(len(ts.getTitleList())),
                'Headline': ts.getTitleList()})

            df = df.append(frame, ignore_index=True)

        return df


    def predictSentiment(self, tickersDF):
        """
        tickersDF --> df with columns = [Ticker, Headline]

        returns --> adds sentiment prediction to df columns = [Date, Ticker, Headline, Prediction]
        """

        dl = DataLoader()

        dl.load_vocab(self.training_data_dir + 'pos', self.training_data_dir + 'neg')

        if self.train_model:
            st = SentimentTrainer(dl.vocab)
            st.train_model(training_data_dir + 'pos', training_data_dir + 'neg')

        sp = SentimentPredictor(dl.vocab)
        sentimentPredictions = sp.predict_sentiment(tickersDF)
        sentimentPredictions.to_csv('../data/sentimentPredictions.csv')

        return sentimentPredictions

    def listToFloat(self, x):

        if np.isnan(x):
            return np.nan
        else:
            return float(x[0][0])

    def getMedianSentiment(self, predictionDF):
        """
            predictionDF --> df with columns = [Date, Ticker, Headline, Prediction]

            Function --> uses z scores to drop outliers 

            returns --> date, median of remaining predictions
        """

        predictionDF['Prediction'] = predictionDF['Prediction'].apply(self.listToFloat)
        predictionDF = predictionDF.dropna()

        z = stats.zscore(predictionDF['Prediction'])
        predictionDF['z score'] = z

        predictionDF.drop(predictionDF[predictionDF['z score'] < -2.5].index, inplace=True)
        predictionDF.drop(predictionDF[predictionDF['z score'] > 2.5].index, inplace=True)


        
        return (predictionDF.iloc[0]['Date'], predictionDF['Prediction'].median())

    def getMonthsSentiment(self, tickers):
        """
        tickers --> dict, ticker : name of company to predict

        function --> scrapes, predicts, and finds median sentiment each day for the last 30 days

        returns --> [dates], [sentiments]
        """
        dates = []
        sentiments = []

        for i in range(self.daysToPredict):
            start = (date.today() - timedelta(days=i))
            end = start
            # end  = start + timedelta(days=1)

            start = start.strftime('%m/%d/%Y')
            end = end.strftime('%m/%d/%Y')

            tickersDF = self.getTitleDF(tickers, start,end)
            predictionDF = self.predictSentiment(tickersDF)
            d, ms = self.getMedianSentiment(predictionDF)

            dates.append(d)
            sentiments.append(ms)

        return (dates, sentiments)


    def writeSentiment(self, df, ticker):
        """
        df --> df of median sentiment each day for last 30 days

        ticker --> company ticker

        function --> saves df to CSV to save time in the future
        """

        start = df.iloc[0]['Dates'].replace('/', '-')
        end = df.iloc[-1]['Dates'].replace('/', '-')
        
        df.to_csv('../data/monthSentiment/' + ticker + '.csv')


    def smoothCurve(self, df):

        x = np.arange(30)
        y = df['Median Sentiment']

        t, c, k = interpolate.splrep(x, y, s=0, k=4)

        N = 100
        xmin, xmax = x.min(), x.max()
        xx = np.linspace(xmin, xmax, N)
        spline = interpolate.BSpline(t, c, k, extrapolate=False)

        return (xx, spline)

        # plt.plot(x, y, 'bo', label='Original points')
        # plt.plot(xx, spline(xx), 'r', label='BSpline')


    def plotSentiment(self, tickerList):
        """
        tickerList --> list of tickers to plot

        function --> plots sentiment using CSV of median sentiment for each day
        """

        # get_ipython().run_line_magic('pylab', '')

        fig, ax = plt.subplots(1,1)
        xcol = 'date'
        ycol = 'Sentiment'

        for ticker in tickerList:

            df = pd.read_csv('../data/monthSentiment/' + ticker + '.csv')

            # xx, spline = smoothCurve(df)
            # ax.plot(xx, spline(xx))

            ax.plot_date(df['Dates'], df['Median Sentiment'], label=ticker, linestyle='-')

        ax.legend()
        ax.grid(True)
        ax.legend(frameon=False)

        ax.set_title('Reader Sentiment Last 30 Days')
        ax.set_xlabel('Date')
        ax.set_ylabel('Sentiment\n 0: Negative 1: Positive')

        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(6))

        # ax.set_yticklabels(['','Bad', '', 'Neutral','', 'Good', ])

        # plt.gcf().autofmt_xdate()
        mplcursors.cursor(hover=True)

        name = '-'.join(tickerList)
        fig.savefig('../visuals/' + name + '-sentiment.png', dpi=fig.dpi)
        # plt.show()


    def plotSmoothCurve(self):
        """
        Originally to smooth the sentiment graph to look like continuous values,
        but was less informative than discrete data points and so is never used. 

        Left in as an ex. of splines and interpolation
        """

        # %matplotlib inline
        df = pd.read_csv('../data/monthSentiment/' + 'MSFT' + '.csv')
        # x = np.arange(30)

        # y = df['Median Sentiment']

        # f = interp1d(x,y, kind='linear')

        x = np.arange(30)
        y = df['Median Sentiment']

        t, c, k = interpolate.splrep(x, y, s=0, k=4)

        N = 100
        xmin, xmax = x.min(), x.max()
        xx = np.linspace(xmin, xmax, N)
        spline = interpolate.BSpline(t, c, k, extrapolate=False)

        # plt.plot(x, y, 'bo', label='Original points')
        plt.plot(xx, spline(xx), 'r', label='BSpline')


        # plt.plot(x, f(x))
        plt.show()


    def main(self):

        # for tickerDict in self.tickerList:
        
        #     dates, sentiments = self.getMonthsSentiment(tickerDict)

        #     df = pd.DataFrame(
        #         {'Dates': dates, 'Median Sentiment': sentiments}).sort_values('Dates')

        #     ticker = list(tickerDict.keys())[0]
            
        #     self.writeSentiment(df, ticker)

        tickerList = [list(dict.keys())[0] for dict in self.tickerList]

        self.plotSentiment(tickerList)


sv = SentimentVizualizer(20, 5, [{'HD': 'Home Depot'}, {'XRX': 'Xerox'}])
sv.main()




