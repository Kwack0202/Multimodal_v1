# ======================================================
import FinanceDataReader as fdr
import pandas_datareader.data as pdr

# ======================================================
# TA-Lib
import talib

# ======================================================
# basic library
import warnings

import openpyxl
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import time
import math
import os
import os.path
import random
import shutil
import glob

# ======================================================
# tqdm|
from tqdm import tqdm

# ======================================================
# datetime
from datetime import timedelta, datetime

# ======================================================
# plotting library
import matplotlib.pyplot as plt

import plotly.graph_objects as go
import plotly.subplots as ms
import plotly.express as px

import mplfinance as fplt
from mplfinance.original_flavor import candlestick2_ohlc, volume_overlay

import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from matplotlib.dates import MonthLocator
from PIL import Image

import seaborn as sns
import cv2
import csv

# plt.rcParams['figure.dpi'] = 150
from IPython.display import clear_output