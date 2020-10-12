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
        self.lines = []
        for p_tag in self.content:
            splitWords = p_tag.string
            if splitWords is None:
                continue
            splitWords = str(splitWords).replace(';','.')
            splitWords = str(splitWords).split('.')
            # splitWords = splitWords.split('.;')
            for line in splitWords:
                self.lines.append(line)

    

    
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

class ProcessContent:

    def __init__(self, content=None):
        if len(content) == 0:
            print("Scraping Error")
            return
        
        self.content = content
        self.clean()
        
        self.debugInFile(self.content)
        self.createConcordance()
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

    def debugInFile(self, content):
        file = open(os.path.join(os.getcwd(), 'SentimentalTrader', 'src', 'webscraping', 'debugFile.txt'), 'w+')
        for line in content:
            file.write(line + '\n')


    
        

session = requests.Session()
#works
# ws = WebScraper(session, "https://finance.yahoo.com/news/lawsuits-filed-against-uco-nnox-143500447.html")

#works
# ws = WebScraper(session, "https://finance.yahoo.com/news/stock-market-news-live-october-12-2020-113806066.html")
ws = WebScraper(session, "https://www.cnn.com/2020/10/12/tech/microsoft-election-ransomware/index.html")
# ws.printMatrix()




pc  = ProcessContent(ws.getContent())
print(pc.getTokens())
print(pc.getVocab())
# ws.printContent()
