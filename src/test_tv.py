from Trade_Visualizer import Trade_Visualizer
import pandas as pd
import numpy as np
from pandas_datareader import data

aapl = data.DataReader('AAPL', 'yahoo', '2010-01-01', '2020-01-01')
act = aapl[['Close']]
pred = act.apply((lambda x : x + 1))
buys = [np.nan] * len(act) 
buy_vals = np.vectorize(int)(np.random.random((40, )) * len(act))
sell_vals = np.vectorize(int)(np.random.random((40, )) * len(act))

for val in buy_vals:
    buys[val] = 'buy'
for val in sell_vals:
    buys[val] = 'sell'

act['Buys'] = buys

# Add the actual prices
tv = Trade_Visualizer()
for i in range(len(act)):
    tv.add_actual('AAPL', act.iloc[i].name, act.iloc[i].loc['Close'])
    tv.add_pred('AAPL', pred.iloc[i].name, pred.iloc[i].loc['Close'])
    tv.add_trade('AAPL', act.iloc[i].name, act.iloc[i].loc['Buys'])
tv.plot_buy_sell()
tv.plot_pred_vs_act()
tv.save_trade_history()
