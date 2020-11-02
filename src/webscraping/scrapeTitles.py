import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import requests
# import nltk
from nltk.corpus import stopwords
from string import punctuation
from collections import Counter
import os
import re
from unidecode import unidecode
from urllib.request import Request, urlopen
from selenium import webdriver
from GoogleNews import GoogleNews


class titleScraper:
    def __init__(self, ticker='None', company='None', start='01/01/20', end='01/08/20', num=0):
        '''
        Inputs: 
                (ticker, company, start, end, num)

                ticker --> Company Ticker

                company --> Company Name

                start --> Start date for scraping

                end --> End date for scraping

                NOTE: dates must be in form mm/dd/yyyy

                num --> number of titles to find

        Function: Call main to scrape articles and write to files
        '''
        self.ticker = ticker
        self.start = start
        self.end = end
        self.titleList = None
        self.coToTicker = {'nikola': 'NKLA', 'microsoft': 'MSFT', 'apple': 'AAPL',
                           'netflix': 'NFLX', 'workday': 'WDAY', 'nvidia': 'NVDA', 'norton': 'NLOK', 'xerox': 'XRX',
                           'hp': 'HPQ', 'micro': 'AMD', 'advanced': 'AMD', 'moderna': 'MRNA', 'peloton': 'PTON', 'home': 'HD', 'depot': 'HD'}
        self.tickerToCo = {value: key for key, value in self.coToTicker.items()}
        self.num = num



    def scrapeTitles(self, num=0):
        '''
            Inputs: 

                num --> finds at least num titles

            Outputs: A list of raw titles
        '''
        found = 0
        titles = []
        end = self.end
        start = self.start

        while found <= num:

            googlenews = GoogleNews(start=start, end=end)
            
            googlenews.search(self.ticker)

            result = googlenews.result()
            df = pd.DataFrame(result)

            self.titleList = df['title'].tolist()
            self.clean()
            self.stripTitleList()
            found += len(self.titleList)

            for t in self.titleList:
                titles.append((t))
            
            start = end
            end = self.extendDate(end, 2)

        self.titleList = titles
        self.end = end


    def clean(self):
        #remove punctuation
        table = str.maketrans('', '', punctuation + "1234567890")

        self.titleList = [unidecode(word).translate(table).strip()
                        for word in self.titleList]

        #strip punctuation
        self.titleList = [
            word for word in self.titleList if not word in punctuation]

        #strip meaningless words
        self.titleList = [word for word in self.titleList if len(word) > 2]

        self.titleList = [word.lower() for word in self.titleList]

        stopWords = set(stopwords.words('english'))

        #strip stopwords
        newContent = []
        for line in self.titleList:
            lines = []
            for word in line.split():
                if word not in stopWords:
                    lines.append(word)

            newContent.append(" ".join(lines))

        self.titleList = newContent

    def stripTitleList(self):

        """
        strips titles that do no contains ticker or company name from self.titleList
        """
        newList = []
        for title in self.titleList:
            if self.ticker.lower() in title or self.tickerToCo[self.ticker] in title:
                newList.append(title)

        self.titleList = newList

    def extendDate(self, date, extension):
        """
        Input: 
        
        date --> date to extend

        extension --> number of days to extend date by 

        Function: Extends date by <extension>
        """

        newEnd = date.split('/')
        days = int(newEnd[1])
        days += extension
        newEnd[1] = str(days)
        return '/'.join(newEnd)

    
    def getTicker(self):
        return self.ticker
    
    def getStartDate(self):
        return self.start

    def getEndDate(self):
        return self.end
    
    def getTitleList(self):
        return self.titleList

    def outputScrape(self):

        scrapesDir = os.getcwd() + '/SentimentalTrader/src/webscraping/' + 'Scrapes'

        if not os.path.exists(scrapesDir):
            os.mkdir(scrapesDir)

        # filepath = scrapesDir + '/' + ts.ticker + '-' + ts.start + '-' + ts.end + '.txt'
        filepath = scrapesDir + '/' + self.ticker + '.txt'


        # if os.path.exists(filepath):
        #     os.remove(filepath)

        file = open(filepath, 'w+')
        

        file.write(self.ticker + '\n'
                + self.start + '\n'
                + self.end + '\n\n')

        for line in self.titleList:
            file.write(line + '\n')

    def main(self):
        self.scrapeTitles(self.num)
        self.outputScrape()



# ts = titleScraper('MSFT', 'Microsoft', '10/10/2020', '10/10/2020', 10)
# ts.main()

ts = titleScraper('PTON', 'Peloton', '10/10/2020', '10/10/2020', 20)
ts.main()
# print(ts.getTitleList())
# ts.outputScrape()
# ts.clean()
# ts.stripTitleList()

# print(ts.ticker)
# print(ts.getStartDate())
# print(ts.getEndDate())
# print(ts.getTitleList())




