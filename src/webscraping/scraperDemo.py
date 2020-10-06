import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import requests
# import nltk
from nltk.corpus import stopwords
from string import punctuation
from collections import Counter
import os

#run this when importing nltk for the first time
# nltk.download()
"""
TO DO:
    - Clean up words by removing whitespace
    - Create and maintain a vocabulary
    - Write a vectorization function to convert words to their vocabulary number
"""

class WebScraper:

    def __init__(self, session, url=""):
        self.url = url
        self.session = session
        if url != "":
            self.loadWebPage()
    
    def loadWebPage(self):
        page = self.session.get(self.url)
        if page.status_code != 200:
            print("Status code %d from %s" % (page.status_code, self.url))
        self.soup = BeautifulSoup(page.content, 'html.parser')
        self.parseContent()
        self.loadIntoNumpyArray()
        
    
    def parseContent(self):
        self.content = list(self.soup.find_all('p'))
        self.words = []
        for p_tag in self.content:
            splitWords = p_tag.string
            if splitWords is None:
                continue
            splitWords = str(splitWords).split(' ')
            for word in splitWords:
                self.words.append(word)

    

    
    def loadIntoNumpyArray(self):
        # while len(self.words) < 2704:
        #     self.words.append("")
        # self.wordMatrix = np.array(self.words).reshape((52, 52))
        self.wordMatrix = np.array(self.words)
    
    def printMatrix(self):
        print(self.wordMatrix)

    def getContent(self):
        return self.wordMatrix

class ProcessContent:

    def __init__(self, content=None):
        if len(content) == 0:
            print("Scraping Error")
            return
        
        self.content = content
        self.clean()
        self.createConcordance()
        self.saveVocab(os.path.join(os.getcwd(),'SentimentalTrader','src','webscraping','vocab.txt'))
    
    def clean(self):
        #remove punctuation
        table = str.maketrans('','',punctuation + "'")
        self.content = [word.translate(table) for word in self.content]

        #strip punctuation 
        self.content = [word for word in self.content if not word in punctuation]

        #strip meaningless words
        self.content = [word for word in self.content if len(word) > 3]

        #strip stopwords
        stopWords = set(stopwords.words('english'))
        self.content = [word for word in self.content if not word in stopWords]

        #strip numbers 
        #can we train based on pos/neg numbers --> this may not be needed
        # self.content = np.array([word for word in self.content if word.isalpha()])
      

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

    
        

session = requests.Session()
#works
ws = WebScraper(session, "https://finance.yahoo.com/news/lawsuits-filed-against-uco-nnox-143500447.html")

#works
# ws = WebScraper(session, "https://www.fool.com/investing/2020/10/03/why-peloton-stock-soared-29-in-september/?source=eptyholnk0000202&utm_source=yahoo-host&utm_medium=feed&utm_campaign=article&yptr=yahoo")
# ws.printMatrix()

pc  = ProcessContent(ws.getContent())
print(pc.getTokens())
print(pc.getVocab())
# ws.printContent()
