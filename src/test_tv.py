from Trade_Visualizer import Trade_Visualizer
import pandas as pd
import numpy as np
from pandas_datareader import data

tv = Trade_Visualizer()
tv.plot_pred_vs_act()
tv.save_trade_history()
