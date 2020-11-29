from trainModel import Trainer
from datetime import datetime
import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from keras import optimizers
from keras.layers.advanced_activations import LeakyReLU
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

class LSTMv2:
    def __init__(self, ticker, raw_data=None, histPoints=30): 
        self.ticker = ticker
        self.trainer = Trainer(ticker, raw_data=raw_data)
        self.histPoints = histPoints
        self.trainer.generateTrainData(histPoints)

        lstmInput = Input(shape=(histPoints, 2), name='lstm_input')
        denseInput = Input(shape=(self.trainer.X_technicals.shape[1],), name='tech_input')

        x = LSTM(histPoints, name='lstm_0')(lstmInput)
        x = Dropout(0.2, name='lstm_dropout_0')(x)
        lstmBranch = Model(inputs=lstmInput, outputs=x)

        y = Dense(20, name='tech_dense_0')(denseInput)
        y = Activation('tanh', name='tech_relu_0')(y)
        y = Dropout(0.2, name='tech_dropout_0')(y)
        techIndicatorsBranch = Model(inputs=denseInput, outputs=y)

        combined = concatenate([lstmBranch.output, techIndicatorsBranch.output], name='concatenate')
        z = Dense(64, activation='sigmoid', name='dense_pooling')(combined)
        z = Dense(1, activation='linear', name='dense_out')(z)

        self.model = Model(inputs=[lstmBranch.input, techIndicatorsBranch.input], outputs=z)
        self.model.compile(optimizer=optimizers.Adam(lr=0.0005), loss='mse')

    def trainModel(self):
        self.model.fit([self.trainer.X_train, self.trainer.X_technicals], self.trainer.Y_train, epochs=50, shuffle=True, validation_split=0.1, batch_size=32)

    def testModel(self, epochs=50):
        trainTestSplit = int(len(self.trainer.raw_data) * 0.80)
        train_ohlc = self.trainer.X_train[:trainTestSplit]
        train_technicals = self.trainer.X_technicals[:trainTestSplit]
        train_Y = self.trainer.Y_train[:trainTestSplit]
        self.model.fit([train_ohlc, train_technicals], train_Y, epochs=epochs, shuffle=True, validation_split=0.1, batch_size=32)

        test_ohlc = self.trainer.X_train[trainTestSplit:]
        test_technicals = self.trainer.X_technicals[trainTestSplit:]
        test_Y = self.trainer.Y_train[trainTestSplit:]
        predictions = self.model.predict([test_ohlc, test_technicals])        
        error = 0
        for (predicted, actual) in zip(predictions, test_Y):
            error += abs(predicted - actual)
        print(error / len(predictions))

        fp = 1
        fn = 1
        tp = 1
        tn = 1
        for i in range(1, len(test_Y)):
            if predictions[i] >= 0 and test_Y[i] >= 0:
                tp += 1
            if predictions[i] >= 0 and test_Y[i] < 0:
                fp += 1
            if predictions[i] < 0 and test_Y[i] < 0:
                tn += 1
            if predictions[i] < 0 and test_Y[i] >= 0:
                fn += 1
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (2 * precision * recall) / (precision + recall)
        print("Precison: ", precision)
        print("Recall: ", recall)
        print("F1: ", f1)
        #with open("../data/epochsVsF1.csv", "a") as fp:
        #    fp.write("%d,%f\n" % (epochs, f1))
    
    def predictNextDay(self):
        changes = self.trainer.raw_data[-1 * self.histPoints:]
        changes = np.hstack((self.trainer.normalizer.transform(changes[:, [0]]), changes[:, [1]]))
        changes = changes.reshape(1, self.histPoints, 2)
        ema12 = self.trainer.expoMovingAvg(12, len(self.trainer.ohlcv))
        ema26 = self.trainer.expoMovingAvg(26, len(self.trainer.ohlcv))
        macd = np.array([ema26 - ema12])
        prediction = self.model.predict([changes, macd])
        return prediction

    def predictDay(self, data):
        changes = np.hstack((self.trainer.normalizer.transform(data[:, [0]]), data[:, [1]]))
        changes = changes.reshape(1, self.histPoints, 2)
        ema12 = self.trainer.expoMovingAvg(12, len(self.trainer.ohlcv))
        ema26 = self.trainer.expoMovingAvg(26, len(self.trainer.ohlcv))
        macd = np.array([ema26 - ema12])
        prediction = self.model.predict([changes, macd])
        return prediction

if __name__ == "__main__":
    ls = LSTMv2('AAPL')
    ls.trainModel()
    #print(ls.predictNextDay())
    ls.testModel(epochs=50)

