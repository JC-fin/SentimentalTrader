from trainModel import Trainer
from datetime import datetime
import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from keras import optimizers
import numpy as np
#np.random.seed(4)
#from tensorflow import set_random_seed
#set_random_seed(4)
import yfinance as yf
import matplotlib.pyplot as plt

class LSTMv2:
    # takes a ticker, a date as a string in the format 'YYYY-MM-DD', and the number of point to predict on
    # also takes in optional raw data that will be used in train_model to limit calls to API
    def __init__(self, ticker, date, histPoints=50, raw_data=None): 
        self.ticker = ticker
        self.trainer = Trainer(ticker, date, raw_data=raw_data)
        self.histPoints = histPoints
        self.trainer.generateTrainData(histPoints)
        #self.trainer.generateTestData(20)

        lstmInput = Input(shape=(histPoints, 5), name='lstm_input')
        denseInput = Input(shape=(self.trainer.X_technicals.shape[1],), name='tech_input')

        x = LSTM(50, name='lstm_0')(lstmInput)
        x = Dropout(0.2, name='lstm_dropout_0')(x)
        lstmBranch = Model(inputs=lstmInput, outputs=x)

        y = Dense(20, name='tech_dense_0')(denseInput)
        y = Activation('relu', name='tech_relu_0')(y)
        y = Dropout(0.2, name='tech_dropout_0')(y)
        techIndicatorsBranch = Model(inputs=denseInput, outputs=y)

        combined = concatenate([lstmBranch.output, techIndicatorsBranch.output], name='concatenate')
        z = Dense(64, activation='sigmoid', name='dense_pooling')(combined)
        z = Dense(1, activation='linear', name='dense_out')(z)

        self.model = Model(inputs=[lstmBranch.input, techIndicatorsBranch.input], outputs=z)
        self.model.compile(optimizer=optimizers.Adam(lr=0.0005), loss='mse')

    def trainModel(self):
        self.model.fit([self.trainer.X_train, self.trainer.X_technicals], self.trainer.Y_train, epochs=50, shuffle=True, validation_split=0.1, batch_size=32)

    def testModel(self):
        trainTestSplit = int(len(self.trainer.raw_data) * 0.80)
        train_ohlc = self.trainer.X_train[:trainTestSplit]
        train_technicals = self.trainer.X_technicals[:trainTestSplit]
        train_Y = self.trainer.Y_train[:trainTestSplit]

        test_ohlc = self.trainer.X_train[trainTestSplit:]
        test_technicals = self.trainer.X_technicals[trainTestSplit:]
        test_Y = self.trainer.Y_train[trainTestSplit:]
        test_Y = self.trainer.y_normalizer.inverse_transform(test_Y)
        predictions = self.model.predict([self.trainer.X_train, self.trainer.X_technicals])
        #print(predictions)
        predictions = self.trainer.y_normalizer.inverse_transform(predictions)
        print(predictions)
        
        error = 0
        for (predicted, actual) in zip(predictions, test_Y):
            error += abs(predicted[0] - actual[0]) / actual[0]
        print(error / len(predictions))

        correct = 0
        for i in range(1, len(test_Y)):
            #print(predictions[i-1:i+1], test_Y[i-1:i+1])
            if predictions[i] >= predictions[i - 1] and test_Y[i] >= test_Y[i - 1]:
                correct += 1
            if predictions[i] < predictions[i - 1] and test_Y[i] < test_Y[i - 1]:
                correct += 1
        print(correct / len(test_Y))
    
    def predictNextDay(self):
        ohlcv = self.trainer.raw_data[-1 * self.histPoints:]
        ohlcv = self.trainer.normalizer.transform(ohlcv)
        ohlcv = ohlcv.reshape(1, self.histPoints, 5)
        technical = np.array(ohlcv[:,3].mean()).reshape((1, 1))
        prediction = self.model.predict([ohlcv, technical])
        return self.trainer.y_normalizer.inverse_transform(prediction)
    
    # Returns the predicted percent change for the day following the last day in the given data
    # Data should be a 2D array with the most recent information in the last row
    def predictDay(self, data):
        if not data.shape == (self.histPoints, 5):
            raise ValueError('predictDay called with ill-sized data. Given: ' + str(data.shape))
        ohlcv = self.trainer.normalizer.transform(data)
        ohlcv = ohlcv.reshape(1, self.histPoints, 5)
        technical = np.array(ohlcv[:,3].mean()).reshape((1, 1))
        prediction = self.model.predict([ohlcv, technical])
        return self.trainer.y_normalizer.inverse_transform(prediction)

if __name__ == "__main__":
    msft = LSTMv2('MSFT', 30)
    msft.trainModel()
    #msft.testModel()
    print(msft.predictNextDay())

