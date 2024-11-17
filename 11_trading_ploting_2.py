from common_imports import *

import pandas as pd
import matplotlib.pyplot as plt

tickers = ['IXIC', 'US500', 'KS11', 'KQ11']
days = ['1day' ,'5day']

years = ['2020','2021','2022']


for year in years:
    for ticker in tickers:
        
        for day in days:
            
            if day == '1day':
                
                if ticker == 'IXIC':
                    Fusion_df = pd.read_csv(f'./Backtesting/Final_results/YOY_{year}/IXIC_Fusion_hard_1day_SVM.csv')
            
                elif ticker == 'US500':
                    Fusion_df = pd.read_csv(f'./Backtesting/Final_results/YOY_{year}/US500_Fusion_hard_1day_xgb.csv')
                
                elif ticker == 'KS11':
                    Fusion_df = pd.read_csv(f'./Backtesting/Final_results/YOY_{year}/KS11_Fusion_hard_1day_SVM.csv')
                
                else:
                    Fusion_df = pd.read_csv(f'./Backtesting/Final_results/YOY_{year}/KQ11_Fusion_soft_1day_xgb.csv')
            
            else:
                if ticker == 'IXIC':
                    Fusion_df = pd.read_csv(f'./Backtesting/Final_results/YOY_{year}/IXIC_Fusion_hard_5day_lgb.csv')
                
                elif ticker == 'US500' :
                    Fusion_df = pd.read_csv(f'./Backtesting/Final_results/YOY_{year}/US500_Fusion_hard_5day_lgb.csv')
                
                elif ticker == 'KS11' :
                    Fusion_df = pd.read_csv(f'./Backtesting/Final_results/YOY_{year}/KS11_Fusion_hard_5day_SVM.csv')
                
                else:
                    Fusion_df = pd.read_csv(f'./Backtesting/Final_results/YOY_{year}/KQ11_Fusion_soft_5day_xgb.csv')
                    
            # 날짜를 인덱스로 설정
            Fusion_df['Date'] = pd.to_datetime(Fusion_df['Date'])
            Fusion_df.set_index('Date', inplace=True)

            # 매수 (Buy)와 매도 (sell) 신호에 대한 인덱스 추출
            buy_signals = Fusion_df[Fusion_df['action'] == 'Buy']
            sell_signals = Fusion_df[Fusion_df['action'] == 'sell']

            # 주식 가격과 신호를 시각화
            plt.figure(figsize=(12, 6))
            plt.tight_layout()
            plt.plot(Fusion_df.index, Fusion_df['Close'], label='Close', color='black', alpha = 0.5)
            plt.scatter(buy_signals.index, buy_signals['Close'], label='Buy', marker='^', color='g', lw=2, s = 50)
            plt.scatter(sell_signals.index, sell_signals['Close'], label='Sell', marker='v', color='r', lw=2, s = 50)

            plt.title(f'Buy Sell Signal : {ticker} {day}')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)

            output_directory = f'./Backtesting/trading_plot/YOY_{year}/'
            os.makedirs(output_directory, exist_ok=True)
        
            plt.savefig(os.path.join(output_directory, f'{ticker}_{day}_signal.png'))
            plt.close()