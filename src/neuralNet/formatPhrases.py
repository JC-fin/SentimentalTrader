import numpy as np
import pandas as pd
import os


if not os.path.exists('../../data/finData'):
    os.mkdir('../../data/finData')

if not os.path.exists('../../data/finData/pos'):
    os.mkdir(os.path.abspath('../../data/finData/pos/'))

if not os.path.exists('../../data/finData/neg'):
    os.mkdir('../../data/finData/neg/')

if not os.path.exists('../../data/finData/neu'):
    os.mkdir('../../data/finData/neu')

numPos = 0
numNeg = 0
numNeu = 0

posDir = os.path.abspath('../../data/finData/pos')
negDir = os.path.abspath('../../data/finData/neg')
neuDir = os.path.abspath('../../data/finData/neu')

def make_text(row):
    global numPos
    global numNeg
    global numNeu
    if row['sent'] == 'positive':
        os.chdir(posDir)
        file_name = 'pos' + str(numPos) + '.txt'
        file = open(file_name,'w')
        file.write(row['phrase'])
        file.close()
        numPos += 1
        print(f'numPos: ', numPos)
    elif row['sent'] == 'negative':
        os.chdir(negDir)
        file_name = 'neg' + str(numNeg) + '.txt'
        file = open(file_name, 'w')
        file.write(row['phrase'])
        file.close()
        numNeg += 1
        print(f'numNeg: ', numNeg)
    elif row['sent'] == 'neutral':
        os.chdir(neuDir)
        file_name = 'neu' + str(numNeu) + '.txt'
        file = open(file_name, 'w')
        file.write(row['phrase'])
        file.close()
        numNeu += 1
        print(f'numNeu: ', numNeu)

def test(row):
    print(row['phrase'])

phrases = pd.read_csv('../../data/all-data.csv', encoding='latin-1', names=['sent','phrase'])

phrases.apply(make_text, axis=1)
