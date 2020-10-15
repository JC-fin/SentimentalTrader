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
        self.model.add(tf.keras.layers.LSTM(20, input_shape=(10, 5), return_sequences=True))
        self.model.add(tf.keras.layers.LSTM(20))
        self.model.add(tf.keras.layers.Dense(1, activation=LeakyReLU(alpha=0.3)))
        #self.model.add(LeakyReLU(alpha=0.3)) tf.nn.tanh
        self.model.compile(optimizer='adam', loss='mean_squared_error')
    
    def trainModel(self):
        #print(self.trainer.Y_train)
        self.model.fit(self.trainer.X_train, self.trainer.Y_train, epochs=150)
    
    def evaluateModel(self):
        print(self.model.evaluate(self.trainer.X_test, self.trainer.Y_test))

    def testModel(self):
        """
        toPredict = yf.Ticker(self.ticker).history(start="2020-01-01", end="2020-10-02")
        toPredict = toPredict['Close']
        actual = np.array(toPredict.iloc[9:])
        toPredict = np.array(toPredict).reshape((19, 10, 1)) / 500
        predicted = self.model.predict(toPredict)
        """
        predicted = self.model.predict(self.trainer.X_test).flatten()
        """[]
        for test in self.trainer.X_test:
            test = test.reshape((1, 10, 5))
            wasPredicted = self.model.predict(test).flatten()
            print(wasPredicted)
            while (wasPredicted == 0):
                wasPredicted = self.model.predict(test).flatten()
            predicted.append(wasPredicted)
        #predicted = self.model.predict(self.trainer.X_test).flatten()
        predicted = np.array(predicted)
        """
        plt.figure(figsize=(15,6))
        #plt.plot(predicted.flatten(), label="predicted")
        plt.plot(self.trainer.Y_test, predicted.flatten(), 'ro')
        plt.ylabel('Predicted')
        plt.xlabel('Actual')
        plt.legend()
        plt.show()

msft = LSTM("MSFT")
msft.trainModel()
#msft.evaluateModel()
msft.testModel()
