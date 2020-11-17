import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import requests
#import nltk
from nltk.corpus import stopwords
from string import punctuation
from collections import Counter
import os
import re
from unidecode import unidecode
from urllib.request import Request, urlopen
from selenium import webdriver
from GoogleNews import GoogleNews

#run this when importing nltk for the first time
#nltk.download()
"""
TO DO:
    - Clean up words by removing whitespace
    - Create and maintain a vocabulary
    - Write a vectorization function to convert words to their vocabulary number
"""

class WebScraper:

    def __init__(self, session, title, url="", ticker='None'):
        if os.path.exists('/Users/MichaelMoschitto/Desktop/CS/301/SentimentalTrader/src/webscraping/rawRequest.txt'):
            os.remove('/Users/MichaelMoschitto/Desktop/CS/301/SentimentalTrader/src/webscraping/rawRequest.txt')
        
        if os.path.exists('/Users/MichaelMoschitto/Desktop/CS/301/SentimentalTrader/src/webscraping/scrape.txt'):
            os.remove('/Users/MichaelMoschitto/Desktop/CS/301/SentimentalTrader/src/webscraping/scrape.txt')

        self.tickers = {'nikola' : 'NKLA', 'microsoft'  : 'MSFT', 'apple' : 'AAPL', 
        'netflix' : 'NFLX', 'workday': 'WDAY', 'nvidia' : 'NVDA', 'norton' : 'NLOK', 'xerox' : 'XRX', 
        'hp': 'HPQ', 'micro': 'AMD', 'advanced': 'AMD', 'moderna' : 'MRNA', 'peloton' : 'PTON', 'home' : 'HD', 'depot' : 'HD'}

        self.companyData = {'ticker' : ticker, 'date': 'None'}
        self.title = title

        # self.options = webdriver.ChromeOptions()


        # self.options.add_argument("user-agent=Mozilla/5.0")
        # self.options.add_argument('--ignore-certificate-errors')
        # self.options.add_argument('--incognito')
        # self.options.add_argument('--headless')
        # driver = webdriver.Chrome("/Users/MichaelMoschitto/Desktop/chromedriver", chrome_options=self.options)

        self.url = url
        self.session = session

        # driver.get(self.url)
        # self.page_source = driver.page_source

        if url != "":
            self.loadWebPage()
    
    def loadWebPage(self):
        page = self.session.get(self.url)
        path = '/Users/MichaelMoschitto/Desktop/CS/301/SentimentalTrader/src/webscraping/Requests/rawRequest-'+str(self.companyData["ticker"])+'-'+str(self.title)+'.txt'
        file = open(path, 'wb')
        file.write(page.content)

        if page.status_code != 200:
            print("Status code %d from %s" % (page.status_code, self.url))
            print('Trying to bypass status code %d' % (page.status_code))

            req = Request(self.url, headers={'User-Agent': 'Mozilla/5.0'})
            self.webpage = urlopen(req).read()

            os.remove(path)
            file = open(path, 'wb')
            file.write(self.webpage)
            self.soup = BeautifulSoup(self.webpage, 'html.parser')

            if self.soup == None:
                return

        else:
            self.soup = BeautifulSoup(page.content, 'html.parser')

        # self.soup = BeautifulSoup(self.page_source, 'html.parser')


        self.parseContent()
        self.loadIntoNumpyArray()

    def getTitles(self, ticker, start, end):
        googlenews = GoogleNews(start=start, end=end)
        googlenews.search(ticker)
        result = googlenews.result()
        df = pd.DataFrame(result)

        return df['title']
    
    def parseContent(self):

        self.parseCompanyName()
        self.parsePublishDate()

        self.content = list(self.soup.find_all('p'))
        # self.content= list(self.soup.find_all('div'))
        self.words = []
        self.lines = []
        for p_tag in self.content:

            #when you take .string on a BS el, if its not a leaf it gives none
            #go through all of the children nodes, might be .strings
            # splitWords = p_tag.string
            splitWords = p_tag.text

            if splitWords is None:
                continue
            splitWords = str(splitWords).replace(';','.')
            splitWords = str(splitWords).split('.')
            # splitWords = splitWords.split('.;')
            for line in splitWords:
                self.lines.append(line)



        # metacontent = list(self.soup.find_all('meta'))
        # file1 = open(os.path.join(os.getcwd(), 'metacontent.txt'), 'w+')
        # for line in metacontent:
        #     file1.write(line + '\n')
        # file1.close()
        # words = []
        # lines = []
        # for tag in self.content:
        #     if tag.has_attr('content'):
        #         splitWords = tag.string
        #     if splitWords is None:
        #         continue
        #     splitWords = str(splitWords).replace(';', '.')
        #     splitWords = str(splitWords).split('.')
        #     # splitWords = splitWords.split('.;')
        #     for line in splitWords:
        #         self.lines.append(line)



    
    def parseCompanyName(self):

        title = self.soup.title.string.split()
        
        for name in title:
            if name.lower() in self.tickers.keys():
                self.companyData['ticker'] = self.tickers[name.lower()]
                return
            elif name in self.tickers.values():
                self.companyData['ticker'] = name
                return


    def parsePublishDate(self):

        #yahoo
        for tag in self.soup.findAll('time'):
            if tag.has_attr('datetime'):
                self.companyData['date'] = pd.to_datetime(tag['datetime'])
                return
        
        #cnn
        for tag in self.soup.find_all('meta'):
            if tag.has_attr('itemprop') and tag['itemprop'] == 'datePublished':
                self.companyData['date'] = pd.to_datetime(tag['content'])
                return                



    

    def loadIntoNumpyArray(self):
        # while len(self.words) < 2704:
        #     self.words.append("")
        # self.wordMatrix = np.array(self.words).reshape((52, 52))
        # self.wordMatrix = np.array(self.lines)
        self.wordMatrix = self.lines
    
    def printMatrix(self):
        print(self.wordMatrix)

    def getContent(self):
        return self.wordMatrix

    def getCompanyData(self):
        return self.companyData

class ProcessContent:

    def __init__(self,title, content=None, data=None):

        if len(content) == 0:
            print("Scraping Error")
            return
        self.title = title
        self.companyData = data
        self.content = content
        self.clean()
        
        self.outputScrape()
        # self.createConcordance()
        # self.saveVocab(os.path.join(os.getcwd(),'SentimentalTrader','src','webscraping','vocab.txt'))
    
    def clean(self):
        #remove punctuation
        table = str.maketrans('','',punctuation + "1234567890")

        self.content = [unidecode(word).translate(table).strip() for word in self.content]

        #strip punctuation 
        self.content = [word for word in self.content if not word in punctuation]

        #strip meaningless words
        self.content = [word for word in self.content if len(word) > 2]

        self.content = [word.lower() for word in self.content]

        
        stopWords = set(stopwords.words('english'))

        #strip stopwords
        newContent = []
        for line in self.content:
            lines = []
            for word in line.split():
                if word not in stopWords:
                    lines.append(word)

            newContent.append(" ".join(lines))

        self.content = newContent


      

    def createConcordance(self):
        #Concordance list ie [ ('price', 34) ...]
        self.vocab = Counter()
        self.vocab.update(self.content)

    def saveVocab(self,fname):
        file = open(fname,'w')
        text = "\n".join(self.content)
        file.write(text)
        file.close()

    def getVocab(self):
        return self.vocab.most_common(50)

    def getTokens(self):
        return self.content

    def outputScrape(self):
       

        path = '/Users/MichaelMoschitto/Desktop/CS/301/SentimentalTrader/src/webscraping/Scrapes/scrape-' + str(self.companyData["ticker"])+'-'+str(self.title)+'.txt'

        if os.path.exists(path):
            os.remove(path)

        file = open(path, 'w+')

        if self.companyData['date'] != 'None':
            file.write("ticker, " + self.companyData['ticker'] + '\n' + 'date, ' + self.companyData['date'].strftime('%Y-%m-%d') + '\n\n')
        else:
            file.write("ticker, " + self.companyData['ticker'] + '\n' + 'date, ' + self.companyData['date'] + '\n\n')

        for line in self.content:
            file.write(line + '\n')


session = requests.Session()

#works
# ws = WebScraper(session, "test", "https://finance.yahoo.com/news/lawsuits-filed-against-uco-nnox-143500447.html")

#works no ticker
# ws = WebScraper(session, 1, "https://finance.yahoo.com/news/stock-market-news-live-october-12-2020-113806066.html", 'TestCO')
# ws = WebScraper(session, "test", "https://www.investors.com/market-trend/stock-market-today/dow-jones-up-walmart-microsoft-power-dow-higher-time-to-buy-box/")

# ws = WebScraper(session, 1, "https://www.investors.com/news/technology/workday-earnings-workday-stock-wday-q2-2020/")
# ws = WebScraper(session, "test", "https://www.investors.com/news/technology/click/netflix-stock-roku-stock-buoyed-positive-data/")
# ws = WebScraper(session, "test", "https://www.investors.com/news/technology/nvidia-stock-soars-data-center-gaming-tailwinds/", 'NVDA')

# ws = WebScraper(session, 1, 'https://www.investors.com/news/nikola-stock-soars-gm-production-partnership-electric-truck-tesla-stock/')

# google docs
ws = WebScraper(
    session, 1, "https://docs.google.com/spreadsheets/d/1l7OiTGWbVkTJz6a1ZHALq0jrfJOPh1dUjNITx4w4854/edit#gid=0", 'Docs')


pc  = ProcessContent(1, ws.getContent(), ws.getCompanyData())


# df = pd.read_csv('/Users/MichaelMoschitto/Desktop/CS/301/SentimentalTrader/src/webscraping/Article List.csv')









# num = 0
# for date, row in df.T.iteritems():
#     # print("Date {}".format(date))
#     for ticker, link in row.iteritems():
#         try:
#             session = requests.Session()
#             ws = WebScraper(session, str(link), str(ticker))
#             pc  = ProcessContent(ws.getContent(), ws.getCompanyData())
#         except:
#             pass
#         break

    

# print(df.loc[:2])

