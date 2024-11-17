# file
import zipfile
import os
import shutil

# sub
from tqdm import tqdm
import random

# basic
import pandas as pd
import numpy as np

# Plot
from PIL import Image
import matplotlib.pyplot as plt

# Tensorflow
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.client import device_lib

# image processing
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# model / neural network
from tensorflow.keras import Sequential, layers, callbacks, backend
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import MinMaxScaler, StandardScaler

# =================================================================================
# Set random seed

seed_value = 42
np.random.seed(seed_value)
random.seed(seed_value)
tf.random.set_seed(seed_value)
# =================================================================================
import xgboost as xgb
import lightgbm as lgb
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# =================================================================================
tickers = ['IXIC', 'US500','KS11', 'KQ11']
days = ['1day', '5day']

for ticker in tickers:
    
    for day in days:
        
        # Train =====================================================
        # LSTM
        LSTM_5 = pd.read_csv(f'./csv/model_results/LSTM_classification_single_results_train/{day}/{ticker}_LSTM_5.csv', index_col=0).reset_index(drop=True)
        LSTM_20 = pd.read_csv(f'./csv/model_results/LSTM_classification_single_results_train/{day}/{ticker}_LSTM_20.csv', index_col=0).reset_index(drop=True)
        LSTM_60 = pd.read_csv(f'./csv/model_results/LSTM_classification_single_results_train/{day}/{ticker}_LSTM_60.csv', index_col=0).reset_index(drop=True)
        LSTM_120 = pd.read_csv(f'./csv/model_results/LSTM_classification_single_results_train/{day}/{ticker}_LSTM_120.csv', index_col=0).reset_index(drop=True)

        ASB_train_LSTM = pd.concat([LSTM_5, LSTM_20['Predicted_results'], LSTM_60['Predicted_results'], LSTM_120['Predicted_results']], axis=1)
        ASB_train_LSTM.columns = ['True_Label', 'LSTM_5', 'LSTM_20', 'LSTM_60', 'LSTM_120']
        
        # VIT
        VIT_5 = pd.read_csv(f'./csv/model_results/VIT_classification_single_results_train/{day}/{ticker}_VIT_5_results.csv', index_col=0).reset_index(drop=True)
        VIT_20 = pd.read_csv(f'./csv/model_results/VIT_classification_single_results_train/{day}/{ticker}_VIT_20_results.csv', index_col=0).reset_index(drop=True)
        VIT_60 = pd.read_csv(f'./csv/model_results/VIT_classification_single_results_train/{day}/{ticker}_VIT_60_results.csv', index_col=0).reset_index(drop=True)
        VIT_120 = pd.read_csv(f'./csv/model_results/VIT_classification_single_results_train/{day}/{ticker}_VIT_120_results.csv', index_col=0).reset_index(drop=True)

        ASB_train_VIT = pd.concat([VIT_5, VIT_20['Predicted_results'], VIT_60['Predicted_results'], VIT_120['Predicted_results']], axis=1)
        ASB_train_VIT.columns = ['True_Label', 'VIT_5', 'VIT_20', 'VIT_60', 'VIT_120']
        
        # Test =====================================================
        # LSTM
        LSTM_5 = pd.read_csv(f'./csv/model_results/LSTM_classification_single_results_test/{day}/{ticker}_LSTM_5.csv', index_col=0).reset_index(drop=True)
        LSTM_20 = pd.read_csv(f'./csv/model_results/LSTM_classification_single_results_test/{day}/{ticker}_LSTM_20.csv', index_col=0).reset_index(drop=True)
        LSTM_60 = pd.read_csv(f'./csv/model_results/LSTM_classification_single_results_test/{day}/{ticker}_LSTM_60.csv', index_col=0).reset_index(drop=True)
        LSTM_120 = pd.read_csv(f'./csv/model_results/LSTM_classification_single_results_test/{day}/{ticker}_LSTM_120.csv', index_col=0).reset_index(drop=True)

        ASB_test_LSTM = pd.concat([LSTM_5, LSTM_20['Predicted_results'], LSTM_60['Predicted_results'], LSTM_120['Predicted_results']], axis=1)
        ASB_test_LSTM.columns = ['True_Label', 'LSTM_5', 'LSTM_20', 'LSTM_60', 'LSTM_120']
        
        # VIT
        VIT_5 = pd.read_csv(f'./csv/model_results/VIT_classification_single_results_test/{day}/{ticker}_VIT_5_results.csv', index_col=0).reset_index(drop=True)
        VIT_20 = pd.read_csv(f'./csv/model_results/VIT_classification_single_results_test/{day}/{ticker}_VIT_20_results.csv', index_col=0).reset_index(drop=True)
        VIT_60 = pd.read_csv(f'./csv/model_results/VIT_classification_single_results_test/{day}/{ticker}_VIT_60_results.csv', index_col=0).reset_index(drop=True)
        VIT_120 = pd.read_csv(f'./csv/model_results/VIT_classification_single_results_test/{day}/{ticker}_VIT_120_results.csv', index_col=0).reset_index(drop=True)

        ASB_test_VIT = pd.concat([VIT_5, VIT_20['Predicted_results'], VIT_60['Predicted_results'], VIT_120['Predicted_results']], axis=1)
        ASB_test_VIT.columns = ['True_Label', 'VIT_5', 'VIT_20', 'VIT_60', 'VIT_120']
        
        # Data merge
        ASB_train = pd.concat([ASB_train_LSTM, ASB_train_VIT.iloc[:, 1:]], axis=1)
        ASB_test = pd.concat([ASB_test_LSTM, ASB_test_VIT.iloc[:, 1:]], axis=1)
        
        os.makedirs('./csv/ASB_csv/soft', exist_ok=True)
         
        ASB_train.to_csv(f'./csv/ASB_csv/soft/{ticker}_{day}_train.csv', index=True)
        ASB_test.to_csv(f'./csv/ASB_csv/soft/{ticker}_{day}_test.csv', index=True)


for ticker in tickers:
    
    for day in days:
        
        # Train =====================================================
        # LSTM
        LSTM_5 = pd.read_csv(f'./csv/model_results/LSTM_classification_binary_results_train/{day}/{ticker}_LSTM_5_train_results_mean.csv', index_col=0).reset_index(drop=True)
        LSTM_20 = pd.read_csv(f'./csv/model_results/LSTM_classification_binary_results_train/{day}/{ticker}_LSTM_20_train_results_mean.csv', index_col=0).reset_index(drop=True)
        LSTM_60 = pd.read_csv(f'./csv/model_results/LSTM_classification_binary_results_train/{day}/{ticker}_LSTM_60_train_results_mean.csv', index_col=0).reset_index(drop=True)
        LSTM_120 = pd.read_csv(f'./csv/model_results/LSTM_classification_binary_results_train/{day}/{ticker}_LSTM_120_train_results_mean.csv', index_col=0).reset_index(drop=True)

        ASB_train_LSTM = pd.concat([LSTM_5, LSTM_20['Predicted_Label'], LSTM_60['Predicted_Label'], LSTM_120['Predicted_Label']], axis=1)
        ASB_train_LSTM.columns = ['True_Label', 'LSTM_5', 'LSTM_20', 'LSTM_60', 'LSTM_120']
        
        # VIT
        VIT_5 = pd.read_csv(f'./csv/model_results/VIT_classification_binary_results_train/{day}/{ticker}_VIT_5_train_results_mean.csv', index_col=0).reset_index(drop=True)
        VIT_20 = pd.read_csv(f'./csv/model_results/VIT_classification_binary_results_train/{day}/{ticker}_VIT_20_train_results_mean.csv', index_col=0).reset_index(drop=True)
        VIT_60 = pd.read_csv(f'./csv/model_results/VIT_classification_binary_results_train/{day}/{ticker}_VIT_60_train_results_mean.csv', index_col=0).reset_index(drop=True)
        VIT_120 = pd.read_csv(f'./csv/model_results/VIT_classification_binary_results_train/{day}/{ticker}_VIT_120_train_results_mean.csv', index_col=0).reset_index(drop=True)

        ASB_train_VIT = pd.concat([VIT_5, VIT_20['Predicted_Label'], VIT_60['Predicted_Label'], VIT_120['Predicted_Label']], axis=1)
        ASB_train_VIT.columns = ['True_Label', 'VIT_5', 'VIT_20', 'VIT_60', 'VIT_120']
        
        # Test =====================================================
        # LSTM
        LSTM_5 = pd.read_csv(f'./csv/model_results/LSTM_classification_binary_results_test/{day}/{ticker}_LSTM_5_test_results_mean.csv', index_col=0).reset_index(drop=True)
        LSTM_20 = pd.read_csv(f'./csv/model_results/LSTM_classification_binary_results_test/{day}/{ticker}_LSTM_20_test_results_mean.csv', index_col=0).reset_index(drop=True)
        LSTM_60 = pd.read_csv(f'./csv/model_results/LSTM_classification_binary_results_test/{day}/{ticker}_LSTM_60_test_results_mean.csv', index_col=0).reset_index(drop=True)
        LSTM_120 = pd.read_csv(f'./csv/model_results/LSTM_classification_binary_results_test/{day}/{ticker}_LSTM_120_test_results_mean.csv', index_col=0).reset_index(drop=True)

        ASB_test_LSTM = pd.concat([LSTM_5, LSTM_20['Predicted_Label'], LSTM_60['Predicted_Label'], LSTM_120['Predicted_Label']], axis=1)
        ASB_test_LSTM.columns = ['True_Label', 'LSTM_5', 'LSTM_20', 'LSTM_60', 'LSTM_120']
        
        # VIT
        VIT_5 = pd.read_csv(f'./csv/model_results/VIT_classification_binary_results_test/{day}/{ticker}_VIT_5_test_results_mean.csv', index_col=0).reset_index(drop=True)
        VIT_20 = pd.read_csv(f'./csv/model_results/VIT_classification_binary_results_test/{day}/{ticker}_VIT_20_test_results_mean.csv', index_col=0).reset_index(drop=True)
        VIT_60 = pd.read_csv(f'./csv/model_results/VIT_classification_binary_results_test/{day}/{ticker}_VIT_60_test_results_mean.csv', index_col=0).reset_index(drop=True)
        VIT_120 = pd.read_csv(f'./csv/model_results/VIT_classification_binary_results_test/{day}/{ticker}_VIT_120_test_results_mean.csv', index_col=0).reset_index(drop=True)

        ASB_test_VIT = pd.concat([VIT_5, VIT_20['Predicted_Label'], VIT_60['Predicted_Label'], VIT_120['Predicted_Label']], axis=1)
        ASB_test_VIT.columns = ['True_Label', 'VIT_5', 'VIT_20', 'VIT_60', 'VIT_120']
        
        # Data merge
        ASB_train = pd.concat([ASB_train_LSTM, ASB_train_VIT.iloc[:, 1:]], axis=1)
        ASB_test = pd.concat([ASB_test_LSTM, ASB_test_VIT.iloc[:, 1:]], axis=1)
        
        os.makedirs('./csv/ASB_csv/hard', exist_ok=True)
        
        ASB_train.to_csv(f'./csv/ASB_csv/hard/{ticker}_{day}_train.csv', index=True)
        ASB_test.to_csv(f'./csv/ASB_csv/hard/{ticker}_{day}_test.csv', index=True)