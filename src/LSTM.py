from trainModel import Trainer
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.layers.advanced_activations import LeakyReLU
import yfinance as yf
import matplotlib.pyplot as plt

class LSTM :
    def __init__(self, ticker):
        self.ticker = ticker
        self.trainer = Trainer(ticker)
        self.trainer.generateTrainData(60)
        self.trainer.generateTestData(60)

        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.LSTM(20, input_shape=(60, 4), return_sequences=True))
        self.model.add(tf.keras.layers.LSTM(20))
        self.model.add(tf.keras.layers.Dense(1, activation=LeakyReLU(alpha=0.3)))
        #self.model.add(LeakyReLU(alpha=0.3)) tf.nn.tanh
        self.model.compile(optimizer='adam', loss='mean_squared_error')
    
    def trainModel(self):
        #print(self.trainer.Y_train)
        self.model.fit(self.trainer.X_train, self.trainer.Y_train, epochs=100)
    
    def evaluateModel(self):
        print(self.model.evaluate(self.trainer.X_test, self.trainer.Y_test))

    def testModel(self):
        predicted = self.model.predict(self.trainer.X_test)
        plt.figure(figsize=(15,6))
        #plt.plot(predicted.flatten(), label="predicted")
        plt.plot(self.trainer.Y_test, label="Actual")
        plt.plot(predicted.flatten(), label='Predicted')
        plt.ylabel('Predicted')
        plt.xlabel('Actual')
        plt.legend()
        plt.show()

msft = LSTM("MSFT")
msft.trainModel()
#msft.evaluateModel()
msft.testModel()
