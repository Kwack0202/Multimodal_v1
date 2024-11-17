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
def train_evaluate_model(model, train_X, train_Y, test_X, test_Y, day, ticker):
    # print(f"Dataset: {dataset_name}")
    
    # Model training
    model.fit(train_X, train_Y)
    
    # Model prediction
    y_pred = model.predict(test_X)
    
    # Model evaluation
    predicted_result_df = pd.DataFrame({'True_Label': test_Y, 'Predicted_results': y_pred})
    
    if not os.path.exists(f'./csv/model_results/Fusion_model_final_results/{day}/'):
                os.makedirs(f'./csv/model_results/Fusion_model_final_results/{day}/')
    predicted_result_df.to_csv(f'./csv/model_results/Fusion_model_final_results/{day}/{ticker}.csv')
    
    # Save classification report to a CSV file
    class_report_df = pd.DataFrame(classification_report(test_Y, y_pred, output_dict=True)).transpose()
    
    if not os.path.exists(f'./csv/model_results/Fusion_model_classification_report/{day}/'):
                os.makedirs(f'./csv/model_results/Fusion_model_classification_report/{day}/')

    class_report_df.to_csv(f'./csv/model_results/Fusion_model_classification_report/{day}/{ticker}.csv', index=True)
    

tickers = ['IXIC', 'US500','KS11', 'KQ11']
days = ['1day', '5day']
votings = ['soft', 'hard']

for ticker in tqdm(tickers):
    
    for day in days:
        
        for voting in votings:
            
            train_df = pd.read_csv(f'./csv/ASB_csv/{voting}/{ticker}_{day}_train.csv', index_col=0)
            test_df = pd.read_csv(f'./csv/ASB_csv/{voting}/{ticker}_{day}_test.csv', index_col=0)
        
            train_X = train_df[['LSTM_5','LSTM_20','LSTM_60','LSTM_120','VIT_5','VIT_20','VIT_60','VIT_120']]
            train_Y = train_df['True_Label']
        
            test_X = test_df[['LSTM_5','LSTM_20','LSTM_60','LSTM_120','VIT_5','VIT_20','VIT_60','VIT_120']]
            test_Y = test_df['True_Label']
        
            ''' xgb_model = xgb.XGBClassifier()
            lgb_model = lgb.LGBMClassifier(verbosity=-1)
            svm_model = SVC(kernel='sigmoid') '''
            
            xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            lgb_model = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=-1
            )
            svm_model = SVC(
                kernel='sigmoid',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            )
        
            # Train and evaluate models
            train_evaluate_model(xgb_model, train_X, train_Y, test_X, test_Y, day, ticker + f'{voting}_xgb')
            train_evaluate_model(lgb_model, train_X, train_Y, test_X, test_Y, day, ticker + f'{voting}_lgb')
            train_evaluate_model(svm_model, train_X, train_Y, test_X, test_Y, day, ticker + f'{voting}_SVM')