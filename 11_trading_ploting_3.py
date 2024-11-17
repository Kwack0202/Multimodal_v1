from common_imports import *


import pandas as pd
import matplotlib.pyplot as plt
import os

# Define colors for LSTM and VIT
lstm_colors = ['lightblue', 'royalblue', 'blue', 'navy']  # Example shades of blue
vit_colors = ['lightcoral', 'indianred', 'firebrick', 'darkred']  # Example shades of red

tickers = ['IXIC', 'US500', 'KS11', 'KQ11']

for ticker in tickers:
    LSTM_5 = pd.read_csv(f'./Backtesting/Final_results/full_period/{ticker}_LSTM_1day_5.csv', index_col=0)
    LSTM_20 = pd.read_csv(f'./Backtesting/Final_results/full_period/{ticker}_LSTM_1day_20.csv', index_col=0)
    LSTM_60 = pd.read_csv(f'./Backtesting/Final_results/full_period/{ticker}_LSTM_1day_60.csv', index_col=0)
    LSTM_120 = pd.read_csv(f'./Backtesting/Final_results/full_period/{ticker}_LSTM_1day_120.csv', index_col=0)
        
    VIT_5 = pd.read_csv(f'./Backtesting/Final_results/full_period/{ticker}_VIT_1day_5.csv', index_col=0)
    VIT_20 = pd.read_csv(f'./Backtesting/Final_results/full_period/{ticker}_VIT_1day_20.csv', index_col=0)
    VIT_60 = pd.read_csv(f'./Backtesting/Final_results/full_period/{ticker}_VIT_1day_60.csv', index_col=0)
    VIT_120 = pd.read_csv(f'./Backtesting/Final_results/full_period/{ticker}_VIT_1day_120.csv', index_col=0)
        
    if ticker == 'IXIC':
        Fusion_df = pd.read_csv('./Backtesting/Final_results/full_period/IXIC_Fusion_hard_1day_SVM.csv')
    
    elif ticker == 'US500':
        Fusion_df = pd.read_csv('./Backtesting/Final_results/full_period/US500_Fusion_hard_1day_xgb.csv')
    
    elif ticker == 'KS11':
        Fusion_df = pd.read_csv('./Backtesting/Final_results/full_period/KS11_Fusion_hard_1day_SVM.csv')
    
    else:
        Fusion_df = pd.read_csv('./Backtesting/Final_results/full_period/KQ11_Fusion_soft_1day_xgb.csv')

    cumulative_profits = {
        'LSTM_5': LSTM_5['Cumulative_Profit'],
        'LSTM_20': LSTM_20['Cumulative_Profit'],
        'LSTM_60': LSTM_60['Cumulative_Profit'],
        'LSTM_120': LSTM_120['Cumulative_Profit'],
        'VIT_5': VIT_5['Cumulative_Profit'],
        'VIT_20': VIT_20['Cumulative_Profit'],
        'VIT_60': VIT_60['Cumulative_Profit'],
        'VIT_120': VIT_120['Cumulative_Profit'],
        'Fusion': Fusion_df['Cumulative_Profit']
    }

    baseline_close = Fusion_df['Close'][0]
    Fusion_df['Close_Relative'] = Fusion_df['Close'] - baseline_close
    
    # 날짜를 인덱스로 설정
    Fusion_df['Date'] = pd.to_datetime(Fusion_df['Date'])
    Fusion_df.set_index('Date', inplace=True)
        
    plt.figure(figsize=(12, 6))
    plt.tight_layout()

    for i, (model, cumulative_profit) in enumerate(cumulative_profits.items()):
        if model.startswith('LSTM'):
            plt.plot(Fusion_df.index, cumulative_profit, label=model, color=lstm_colors[i % len(lstm_colors)], alpha=0.2)
        elif model.startswith('VIT'):
            plt.plot(Fusion_df.index, cumulative_profit, label=model, color=vit_colors[i % len(vit_colors)], alpha=0.2)
        elif model == 'Fusion':
            plt.plot(Fusion_df.index, cumulative_profit, label=model, color='purple', linewidth=3)

    plt.plot(Fusion_df.index, Fusion_df['Close_Relative'], label='Buy & Hold', linestyle='--')

    plt.title(f'Cumulative Profit Comparison: {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Profit / Close Relative')
    plt.legend()
    plt.grid(True)

    output_directory = './Backtesting/trading_plot/'
    os.makedirs(output_directory, exist_ok=True)

    plt.savefig(os.path.join(output_directory, f'{ticker}_1day_cum.png'))
    plt.close()