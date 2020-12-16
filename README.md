# SentimentalTrader
Data 301 Quarter-long Project: A trading bot combining sentiment analysis on web-scraped articles and quantitative analysis of stock trends in order to enact stock trades.

## Trading Bot
The trading bot uses the results from both the quantitative and the sentiment models for a particular stock to make a decision of whether to buy or sell the stock each day. The bot has been trading on a paper account using the Alpaca API, and through its first 2 weeks trading, earned about a 4% ROI.

## Models
### Quantitative Analysis
#### Quantitative Data Collection and API Interaction
The data for the quantitative model is gathered using a wrapper for Twelve Data, an online purveyor of stock metrics. In important part of the data collected is the Moving Average Convergence Divergence (macd), along with percent change and volume.

#### Quantitative Analysis LSTM
The model consists of two input branches, a Long Short Term Memory (LSTM) network and a standard sequential neural netowrk with one hidden layer. These two branches are then concatenated with a pooling layer. This model achieved F1 scores as high as 0.71 for certain stocks when evaluated on future stock prices.

#### Quantitative Vizualization
Visuals were created for all 12 stocks being tracked. To see other prediction visualizations, see the visuals directory.
![Netflix Prediction Chart](https://github.com/d-mooers/SentimentalTrader/blob/master/visuals/nflx_prediction.png)

### Sentiment Analysis
#### Webscraping
The Google News API is used to search and collect headlines for all of the companies being tracked. These headlines are filtered by company name and/or ticker, and then input into the model to get a sentiment prediction for each company.

#### Sentiment Analysis CNN
The model is a Convolutional Neural Network (CNN) with an initial embedding layer followed by a single convolutional layer, which is then pooled and flattened for the two dense layers, the first using a rectified linear activation function and the final layer using a sigmoid. The model was trained using a data set of 1,967 financial headlines with labeled positive or negative sentiment. When evaluated, the model achieved an F1 score of 0.87 for positive articles.

Given headlines that imply a positive outlook for a company, the model should output a prediction greater than 0.5. Likewise, headlines that imply a negative outlook should receive predictions less than 0.5:
| Input | Output|
| --- | --- |
| "Apple Is Reportedly Working on Custom Silicon for Apple Car" | 0.96578010 |
| "Tesla Beats on Profit, Reaffirms Goal of 500,000 Deliveries" | 0.98735297 |
| "Struggling Nikola Stock Still Has Further to Fall" | 0.06987518 |
| "Purdue Pharma to Plead Guilty, Pay $8.3 Billion Over Opiods" | 0.114830494 |
