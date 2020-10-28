from TwelveDataWrapper import TwelveDataWrapper as tdw
import numpy as np
import pandas as pd
import tensorflow as tf


# This is a method returns a tuple of four numpy arrays: Input training data, output training data, input testing
# data and output testing data. This data is generated given a list of tickers, the date for which the tickers are
# being trained/predicted, the interval on which we will collect data to train the network, the number of points per
# training set, the number of training sets per ticker, and the proportion of the final product that is devoted to testing
def gen_train_test (tickers, test_time=None, interval="1day", numpoints=10, numsets=100, training_proportion=0.8):
    td = tdw()
    X = np.full(shape=(0, numpoints, 5), fill_value=0, dtype='float')
    Y = np.full(shape=(0,), fill_value = 0, dtype='float')
    for ticker in tickers:
        stock_data = td.time_series(tickers=[ticker], interval=interval,
            outputSize=(1+numpoints) * numsets, end_date=test_time)
        print(stock_data)
        test_rest = np.array(stock_data.iloc[stock_data.index % (numpoints + 1) != 0][['open', 'high', 'low', 'close', 'volume']])
        test_rest = test_rest.reshape((numsets, numpoints, 5))
        #Pure closing values
        test_closing = np.array(stock_data.iloc[stock_data.index % (numpoints + 1) == 0]['close'])
        print(test_closing)
        #Percent change of closing values
        test_closing = test_closing / np.array(stock_data.iloc[stock_data.index % (numpoints + 1) == 1]['close'])
        X = np.concatenate([X, test_rest])
        Y = np.concatenate([Y, test_closing])
    #Randomly selects ((1 - training_proportion) * (number of tickers * numsets)) X and Y indicies and makes them testing data
    testing_indicies = np.random.random((int((1 - training_proportion) * len(tickers) * numsets))) * len(tickers) * numsets
    testing_indicies //= 1
    testing_indicies = np.vectorize(int)(testing_indicies)
    print("Testing length: ", len(testing_indicies))
    X_test = X[testing_indicies]
    np.delete(X, testing_indicies)
    Y_test = Y[testing_indicies]
    np.delete(Y, testing_indicies)
    print("Test shape: ", Y_test.shape)
    return (X, Y, X_test, Y_test)

# Uses data from above gen_train_test method to train and test a network
def train_model():
    test_and_train = gen_train_test(['MSFT'])
    X_train = test_and_train[0]
    Y_train = test_and_train[1]
    X_test = test_and_train[2]
    Y_test = test_and_train[3]

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(20, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(tf.keras.layers.LSTM(20))
    model.add(tf.keras.layers.Dense(1, activation =tf.nn.relu))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, Y_train, epochs=250)
    print("Evaluation: ")
    model.evaluate(X_test, Y_test)
    print("Prediction: ")
    predict = model.predict(X_test)
    while predict[0] == 0:
          predict = model.predict(X_test)
    print (predict)
gen_train_test(['MSFT'])
