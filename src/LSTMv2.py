from trainModel import Trainer
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.layers.advanced_activations import LeakyReLU
import yfinance as yf
import matplotlib.pyplot as plt

class LSTMv2:

    def __init__(self, ticker): 
        self.ticker = ticker
        self.trainer = Trainer(ticker)
        self.trainer.generateTrainData(20)
        self.trainer.generateTestData(20)

