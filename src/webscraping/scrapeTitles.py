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
import datetime


class TitleScraper:
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

            How: Makese API call to Google News
                Cleans title list down to just words
                Strips out any titles that do not contain ticker
        '''
        found = 0
        titles = []
        end = self.end
        start = self.start

        # max start can be reduced by is 7 days
        tries = 0

        while found <= num:

            if not self.validDates() or tries > 7:
                break
        

            googlenews = GoogleNews(start=start, end=end)
            
            googlenews.search(self.ticker)

            result = googlenews.result()
            if len(result) == 0:
                break
            df = pd.DataFrame(result)

            if len(df) > 0:
                self.titleList = df['title'].tolist() + df['desc'].tolist()

            self.clean()
            self.stripTitleList()
            

            for t in self.titleList:
                if t not in titles:
                    titles.append((t))
                    found += 1
            
            start = self.reduceDate(start, 1)
            
            tries += 1
           
        self.start = start
        self.titleList = titles[:num]


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
        Checks that either the ticker or company name is in the title before adding it to new list
            Does not add duplicates
        """
        newList = []
        for title in self.titleList:
            if self.ticker.lower() in title or self.tickerToCo[self.ticker] in title:

                if not title in newList:
                    newList.append(title)

        self.titleList = newList

    def reduceDate(self, date, reduction):
        """
        Input: 
        
        date --> date to extend

        reduction --> number of days to extend date by 

        Function: Extends date by <reduction>
        """

        d = datetime.datetime.strptime(date, "%m/%d/%Y")
        d = d - datetime.timedelta(days=reduction)
        return d.strftime("%m/%d/%Y")

    def validDates(self, date=None):


        """
            Checks to make sure that self.start and self.end are not in the future

            Output: 
                True: if both dates are at least 1 day in the past
                False: if either date is present or in the future

        """
        
        start = datetime.datetime.strptime(self.start, "%m/%d/%Y").date()
        if date == None:
            end = datetime.datetime.strptime(self.end, "%m/%d/%Y").date()
        else:
            end = datetime.datetime.strptime(date, "%m/%d/%Y").date()

        today = datetime.date.today()


        return start <= today and end <= today

    def getTicker(self):
        return self.ticker
    
    def getStartDate(self):
        return self.start

    def getEndDate(self):
        return self.end
    
    def getTitleList(self):
        return self.titleList

    def outputScrape(self):

        scrapesDir = os.path.dirname(os.path.abspath(__file__)) + '/Scrapes'

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




# ts.main()

# ts = TitleScraper('PTON', 'Peloton', '11/15/2020', '10/10/2010', 100)


# ts.main()
# print(ts.getTitleList())
# ts.outputScrape()
# ts.clean()
# ts.stripTitleList()

# print(ts.ticker)
# print(ts.getStartDate())
# print(ts.getEndDate())
# print(ts.getTitleList())




