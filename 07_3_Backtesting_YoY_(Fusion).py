from common_imports import *

tickers = ['IXIC', 'US500','KS11', 'KQ11']
days = ['1day', '5day']
models = ['lgb', 'SVM', 'xgb']
votings = ['soft', 'hard']

years= ['2020', '2021', '2022']

opposite_count = 2

for ticker in tickers:
    
    for day in days:
        
        for model in models:
            
            for voting in votings:
                
                trading_data = pd.read_csv(f'./csv/model_results/Fusion_model_final_results/{day}/{ticker}{voting}_{model}.csv', index_col=0).reset_index(drop=True)
                
                for year in years:
                    if year == '2020':
                        if ticker == 'KS11':
                            trading_df = trading_data.iloc[0:248].reset_index(drop=True)
                        elif ticker == 'KQ11':
                            trading_df = trading_data.iloc[0:248].reset_index(drop=True)
                        elif ticker == 'IXIC':
                            trading_df = trading_data.iloc[0:253].reset_index(drop=True)
                        elif ticker == 'US500':
                            trading_df = trading_data.iloc[0:253].reset_index(drop=True)
                        
                    elif year =='2021':
                        if ticker == 'KS11':
                            trading_df = trading_data.iloc[248:496].reset_index(drop=True)
                        elif ticker == 'KQ11':
                            trading_df = trading_data.iloc[248:496].reset_index(drop=True)
                        elif ticker == 'IXIC':
                            trading_df = trading_data.iloc[253:505].reset_index(drop=True)
                        elif ticker == 'US500':
                            trading_df = trading_data.iloc[253:505].reset_index(drop=True)
                        
                    else:
                        if ticker == 'KS11':
                            trading_df = trading_data.iloc[496:].reset_index(drop=True)
                        elif ticker == 'KQ11':
                            trading_df = trading_data.iloc[496:].reset_index(drop=True)
                        elif ticker == 'IXIC':
                            trading_df = trading_data.iloc[505:].reset_index(drop=True)
                        elif ticker == 'US500':
                            trading_df = trading_data.iloc[505:].reset_index(drop=True)

                    
                    #=======================================================================================
                    # 시작 전 기본설정된 매매 신호는 'No action'이라고 가정합니다.
                    action = "No action"
                    counter = 0
                    initial_position_set = False

                    for i in range(len(trading_df)):
                        curr_pos = trading_df.loc[i, 'Predicted_results']
                        if i == 0:
                            prev_pos = 0
                        else:
                            prev_pos = trading_df.loc[i-1, 'Predicted_results']

                        if not initial_position_set:
                            if curr_pos == 0:
                                action = "No action"
                            else:
                                action = "Buy"
                                initial_position_set = True
                        else:
                            last_action = trading_df.loc[i-1, 'action']

                            if last_action == "sell":
                                if curr_pos == 0:
                                    action = "No action"
                                    initial_position_set = False
                                else:
                                    action = "Buy"
                                    counter = 0
                            else:
                                if curr_pos == 1:
                                    action = "Holding"
                                    counter = 0
                                else:
                                    counter += 1
                                    if counter == opposite_count:
                                        action = "sell"
                                        counter = 0
                                    else:
                                        action = "Holding"
                    
                        if i == len(trading_df) - 1:
                            action = "sell"
                    
                        trading_df.loc[i, 'action'] = action
                    
                    if not os.path.exists(f'./Backtesting/trading_signal/YOY_{year}/Fusion/'):
                        os.makedirs(f'./Backtesting/trading_signal/YOY_{year}/Fusion/')
               
                    trading_df.to_csv(f'./Backtesting/trading_signal/YOY_{year}/Fusion/{ticker}_{voting}_{day}_{model}_signal.csv', index=True)