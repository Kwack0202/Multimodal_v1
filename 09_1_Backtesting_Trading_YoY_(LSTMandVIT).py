from common_imports import *

tickers = ['IXIC', 'US500','KS11', 'KQ11']
time_steps = ['5', '20', '60', '120']
days = ['1day', '5day']
models = ['LSTM', 'VIT']

years= ['2020', '2021', '2022']

for ticker in tickers:
    
    for time_step in time_steps:
        
        for day in days:
            
            for model in models:
                
                for year in years:
                    df = pd.read_csv(f"./Backtesting/trading_signal/YOY_{year}/{model}/{ticker}_{day}_signal.csv", index_col=0)
                    
                    
                    df['Date'] = pd.to_datetime(df['Date'])
                    
                    df = df[(df['Date'] >= f'{year}-01-01') & (df['Date'] <= f'{year}-12-31')]
                    
                    df = df.reset_index(drop=True)
                
                    # 새로운 데이터프레임 생성
                    new_data = {
                        'Date': [],
                        'Sell_Signal_Profit': [],
                        'Sell_Signal_Return': [],
                        'Cumulative_Profit': []
                    }

                    buy_price = None
                    current_profit = 0
                    current_profit_ratio = 0 

                    for index, row in df.iterrows():
                        if row[f'action_{time_step}'] == 'Buy':
                            buy_price = row['Close']
                            new_data['Date'].append(row['Date'])
                            new_data['Sell_Signal_Profit'].append(0)
                            new_data['Sell_Signal_Return'].append(0)
                            new_data['Cumulative_Profit'].append(current_profit)
                            
                        elif row[f'action_{time_step}'] == 'sell' and buy_price is not None:
                            if index + 1 < len(df):
                                next_row = df.iloc[index + 1]  # 다음 행을 가져오기
                                sell_price = next_row['Open']
                                profit = sell_price - buy_price
                                current_profit += profit
                                return_ = profit / buy_price * 100
                                current_profit_ratio += return_
                                new_data['Date'].append(row['Date'])
                                new_data['Sell_Signal_Profit'].append(profit)
                                new_data['Sell_Signal_Return'].append(return_)
                                new_data['Cumulative_Profit'].append(current_profit)
                            else:
                                # 다음 행이 없는 경우 해당 행의 Close로 매도
                                sell_price = row['Close']
                                profit = sell_price - buy_price
                                current_profit += profit
                                return_ = profit / buy_price * 100
                                current_profit_ratio += return_
                                new_data['Date'].append(row['Date'])
                                new_data['Sell_Signal_Profit'].append(profit)
                                new_data['Sell_Signal_Return'].append(return_)
                                new_data['Cumulative_Profit'].append(current_profit)
                                
                        else:
                            new_data['Date'].append(row['Date'])
                            new_data['Sell_Signal_Profit'].append(0)
                            new_data['Sell_Signal_Return'].append(0)
                            new_data['Cumulative_Profit'].append(current_profit)

                    # 새로운 데이터프레임 생성
                    new_df = pd.DataFrame(new_data)

                    # "Date" 열을 기준으로 두 데이터프레임 병합
                    merged_df = pd.merge(df, new_df, on='Date', how='outer')
                    
                    merged_df = merged_df[['Date', 'Open', 'High', 'Low', 'Close', f'Predicted_results_{time_step}', f'action_{time_step}', 'Sell_Signal_Profit', 'Sell_Signal_Return', 'Cumulative_Profit']]
                    
                    if not os.path.exists(f'./Backtesting/trading_results/YOY_{year}/{model}/'):
                        os.makedirs(f'./Backtesting/trading_results/YOY_{year}/{model}/')
                        
                    merged_df.to_csv(f'./Backtesting/trading_results/YOY_{year}/{model}/{ticker}_{day}_{time_step}_signal.csv', index=True)