from common_imports import *

time_steps = ['5', '20', '60', '120']
tickers = ['IXIC', 'US500','KS11', 'KQ11']
days = ['1day', '5day']

opposite_count = 2

for ticker in tickers:
    for day in days:
        if ticker == 'KS11':
            signal_data = pd.read_csv('./csv/TA_csv/KS11.csv').iloc[-740:, 0:5].reset_index(drop=True)
        if ticker == 'KQ11':
            signal_data = pd.read_csv('./csv/TA_csv/KQ11.csv').iloc[-740:, 0:5].reset_index(drop=True)
        if ticker == 'IXIC':
            signal_data = pd.read_csv('./csv/TA_csv/IXIC.csv').iloc[-756:, 0:5].reset_index(drop=True)
        if ticker == 'US500':
            signal_data = pd.read_csv('./csv/TA_csv/US500.csv').iloc[-756:, 0:5].reset_index(drop=True)
        
        df_5 = pd.read_csv(f'./csv/model_results/VIT_classification_binary_results_test/{day}/{ticker}_VIT_5_test_results_train_mean.csv', index_col=0).reset_index(drop=True)
        df_20 = pd.read_csv(f'./csv/model_results/VIT_classification_binary_results_test/{day}/{ticker}_VIT_20_test_results_train_mean.csv', index_col=0).reset_index(drop=True)
        df_60 = pd.read_csv(f'./csv/model_results/VIT_classification_binary_results_test/{day}/{ticker}_VIT_60_test_results_train_mean.csv', index_col=0).reset_index(drop=True)
        df_120 = pd.read_csv(f'./csv/model_results/VIT_classification_binary_results_test/{day}/{ticker}_VIT_120_test_results_train_mean.csv', index_col=0).reset_index(drop=True)
        
        ASB_test = pd.concat([df_5['Predicted_Label'], df_20['Predicted_Label'], df_60['Predicted_Label'], df_120['Predicted_Label']], axis=1)
        ASB_test.columns = ['Predicted_results_5', 'Predicted_results_20', 'Predicted_results_60', 'Predicted_results_120']
        
        trading_data = pd.concat([signal_data, ASB_test], axis=1)
    
        for time_step in time_steps:
            action = "No action"
            counter = 0
            initial_position_set = False

            for i in range(len(trading_data)):
                curr_pos = trading_data.loc[i, f'Predicted_results_{time_step}']
                if i == 0:
                    prev_pos = 0
                else:
                    prev_pos = trading_data.loc[i-1, f'Predicted_results_{time_step}']

                if not initial_position_set:
                    if curr_pos == 0:
                        action = "No action"
                    else:
                        action = "Buy"
                        initial_position_set = True
                else:
                    last_action = trading_data.loc[i-1, f'action_{time_step}']

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
                
                trading_data.loc[i, f'action_{time_step}'] = action
                
            output_dir = './Backtesting/trading_signal/full_period/VIT'
            os.makedirs(output_dir, exist_ok=True) 
            trading_data.to_csv(f'./Backtesting/trading_signal/full_period/VIT/{ticker}_{day}_signal.csv', index=True)