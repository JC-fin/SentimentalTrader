import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import requests
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
        #print(self.content)
        self.words = []
        for p_tag in self.content:
            splitWords = p_tag.string
            if splitWords is None:
                continue
            splitWords = str(splitWords).split(' ')
            for word in splitWords:
                self.words.append(word)
    
    def loadIntoNumpyArray(self):
        while len(self.words) < 2704:
            self.words.append("")
        self.wordMatrix = np.array(self.words).reshape((52, 52))
    
    def printMatrix(self):
        print(self.wordMatrix)

session = requests.Session()
ws = WebScraper(session, "https://finance.yahoo.com/news/lawsuits-filed-against-uco-nnox-143500447.html")
ws.printContent()