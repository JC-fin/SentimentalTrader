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
        self.trainer.generateTrainData(10)
        self.trainer.generateTestData(10)

        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.LSTM(20, input_shape=(10, 3), return_sequences=True))
        self.model.add(tf.keras.layers.LSTM(20))
        self.model.add(tf.keras.layers.Dense(1, activation=LeakyReLU(alpha=0.3)))
        #self.model.add(LeakyReLU(alpha=0.3)) tf.nn.tanh
        self.model.compile(optimizer='adam', loss='mean_squared_error')
    
    def trainModel(self):
        print(self.trainer.Y_train)
        print(self.trainer.X_train)
        self.model.fit(self.trainer.X_train, self.trainer.Y_train, epochs=100)
    
    def evaluateModel(self):
        print(self.model.evaluate(self.trainer.X_test, self.trainer.Y_test))

    def testModel(self):
        predicted = self.model.predict(self.trainer.X_test)
        plt.figure(figsize=(15,6))
        plt.plot(predicted.flatten(), label="predicted")
        plt.plot(self.trainer.Y_test, label='actual')
        #predicted = predicted.flatten()
        #correctness = abs(predicted - self.trainer.Y_test) / self.trainer.Y_test
        #print(correctness.mean())
        #print(right.mean())
        #print(((predicted / abs(predicted)) == self.trainer.Y_test).mean())
        #print(self.trainer.Y_test)
        #plt.ylabel('Predicted')
        plt.xlabel('Actual')
        plt.legend()
        plt.show()

msft = LSTM("MSFT")
msft.trainModel()
msft.evaluateModel()
msft.testModel()
