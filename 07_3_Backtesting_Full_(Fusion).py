from common_imports import *

tickers = ['IXIC', 'US500', 'KS11', 'KQ11']
days = ['1day', '5day']
models = ['lgb', 'SVM', 'xgb']
votings = ['soft', 'hard']

opposite_count = 2

for ticker in tickers:
    
    for day in days:
        
        for model in models:
            
            for voting in votings:
                
                if ticker == 'KS11':
                    signal_data = pd.read_csv('./csv/TA_csv/KS11.csv').iloc[-740:, 0:5].reset_index(drop=True)
                if ticker == 'KQ11':
                    signal_data = pd.read_csv('./csv/TA_csv/KQ11.csv').iloc[-740:, 0:5].reset_index(drop=True)
                if ticker == 'IXIC':
                    signal_data = pd.read_csv('./csv/TA_csv/IXIC.csv').iloc[-756:, 0:5].reset_index(drop=True)
                if ticker == 'US500':
                    signal_data = pd.read_csv('./csv/TA_csv/US500.csv').iloc[-756:, 0:5].reset_index(drop=True)
            
                trading_data = pd.read_csv(f'./csv/model_results/Fusion_model_final_results/{day}/{ticker}{voting}_{model}.csv', index_col=0).reset_index(drop=True)

                # 기본설정된 매매 신호는 'No action'이라고 가정합니다.
                action = "No action"
                counter = 0
                initial_position_set = False

                for i in range(len(trading_data)):
                    curr_pos = trading_data.loc[i, 'Predicted_results']
                    if i == 0:
                        prev_pos = 0
                    else:
                        prev_pos = trading_data.loc[i-1, 'Predicted_results']

                    if not initial_position_set:
                        if curr_pos == 0:
                            action = "No action"
                        else:
                            action = "Buy"
                            initial_position_set = True
                    else:
                        last_action = trading_data.loc[i-1, 'action']

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
                
                    if i == len(trading_data) - 1:
                        action = "sell"
                
                    trading_data.loc[i, 'action'] = action

                output_dir = './Backtesting/trading_signal/full_period/Fusion'
                os.makedirs(output_dir, exist_ok=True) 
                trading_data.to_csv(f'./Backtesting/trading_signal/full_period/Fusion/{ticker}_{voting}_{day}_{model}_signal.csv', index=True)