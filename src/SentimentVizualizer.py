import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


class SentimentVizualizer:

    def __init__(self):
        self.trade_hist_path = '../data/trade_history.json'
        self.visuals_path = '../visuals/'
