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
        self.trainer.generateTrainData(20)
        self.trainer.generateTestData(20)

        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.LSTM(20, input_shape=(20, 2), return_sequences=True))
        self.model.add(tf.keras.layers.LSTM(20))
        self.model.add(tf.keras.layers.Dense(2, activation='softmax'))
        #self.model.add(LeakyReLU(alpha=0.3)) tf.nn.tanh
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    def trainModel(self):
        self.model.fit(self.trainer.X_train, self.trainer.Y_train, epochs=50)
    
    def predict(self, x_data):
        return self.model.predict(x_data)
    
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
    
    def movementPrediction(self):
        predicted = self.model.predict(self.trainer.X_test)
        right = 0
        for i in range(len(predicted)):
            expected = self.trainer.Y_test[i]
            if (expected[0] and predicted[i][0] > 0.5) :
                right += 1
            if (expected[1] and predicted[i][1] > 0.5) :
                right += 1
        print(right / len(predicted))

    def extractFeatures(self, data):
        return data

if __name__ == '__main__':
    msft = LSTM("MSFT")
    msft.trainModel()
    msft.evaluateModel()
    msft.movementPrediction()
    #msft.testModel()
