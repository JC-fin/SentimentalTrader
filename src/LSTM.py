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
        self.trainer.generateTrainData(30)
        self.trainer.generateTestData(30)

        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.LSTM(20, input_shape=(30, 5), return_sequences=True))
        self.model.add(tf.keras.layers.LSTM(20))
        #self.model.add(tf.keras.layers.LSTM(20, activation='relu'))
        self.model.add(tf.keras.layers.Dense(3, activation='sigmoid'))
        #self.model.add(LeakyReLU(alpha=0.3))
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    def trainModel(self, num_epochs=85):
        print(self.trainer.X_train)
        print(self.trainer.Y_train)
        self.model.fit(self.trainer.X_train, self.trainer.Y_train, epochs=num_epochs,  batch_size=128)

    def evaluateModel(self):
        print("EVALUATE")
        return self.model.evaluate(self.trainer.X_test, self.trainer.Y_test)

    def testModel(self):
        print("PREDICT")
        print(self.model.predict(self.trainer.X_test))
        #interests = [65, 70, 75, 80, 85, 90]
        #results = []
        #for i in interests:
        #    sum = 0
        #    for j in range(10):
        #        self.trainModel(i)
        #        sum += self.evaluateModel()[1]
        #        self = LSTM("MSFT")
        #    results.append(sum / 10)
        #print(dict(zip(interests, results)))
        #print(self.model.predict(self.trainer.X_test))
        #plt.figure(figsize=(15,6))
        #plt.plot(predicted.flatten(), label="predicted")
        #plt.plot(self.trainer.Y_test, label='actual')
        #predicted = predicted.flatten()
        #correctness = abs(predicted - self.trainer.Y_test) / self.trainer.Y_test
        #print(correctness.mean())
        #print(right.mean())
        #print(((predicted / abs(predicted)) == self.trainer.Y_test).mean())
        #print(self.trainer.Y_test)
        #plt.ylabel('Predicted')
        #plt.xlabel('Actual')
        #plt.legend()
        #plt.show()

msft = LSTM("MSFT")
msft.trainModel()
msft.evaluateModel()
msft.testModel()
