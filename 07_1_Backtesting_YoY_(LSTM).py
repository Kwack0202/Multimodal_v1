from common_imports import *

time_steps = ['5', '20', '60', '120']
tickers = ['IXIC', 'US500','KS11', 'KQ11']
days = ['1day', '5day']

years= ['2020', '2021', '2022']

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
        
        df_5 = pd.read_csv(f'./csv/model_results/LSTM_classification_binary_results_test/{day}/{ticker}_LSTM_5_test_results_mean.csv', index_col=0)
        df_20 = pd.read_csv(f'./csv/model_results/LSTM_classification_binary_results_test/{day}/{ticker}_LSTM_20_test_results_mean.csv', index_col=0)
        df_60 = pd.read_csv(f'./csv/model_results/LSTM_classification_binary_results_test/{day}/{ticker}_LSTM_60_test_results_mean.csv', index_col=0)
        df_120 = pd.read_csv(f'./csv/model_results/LSTM_classification_binary_results_test/{day}/{ticker}_LSTM_120_test_results_mean.csv', index_col=0)
        
        ASB_test = pd.concat([df_5['Predicted_Label'], df_20['Predicted_Label'], df_60['Predicted_Label'], df_120['Predicted_Label']], axis=1)
        ASB_test.columns = ['Predicted_results_5', 'Predicted_results_20', 'Predicted_results_60', 'Predicted_results_120']
        
        trading_data = pd.concat([signal_data, ASB_test], axis=1)

        trading_data['Date'] = pd.to_datetime(trading_data['Date'])
                
        for year in years:

            # 2022년에 해당하는 데이터만 선택
            trading_df = trading_data[(trading_data['Date'] >= f'{year}-01-01') & (trading_data['Date'] <= f'{year}-12-31')]
            
            trading_df = trading_df.reset_index(drop=True)
        
            for time_step in time_steps:
                action = "No action"
                counter = 0
                initial_position_set = False

                for i in range(len(trading_df)):
                    curr_pos = trading_df.loc[i, f'Predicted_results_{time_step}']
                    if i == 0:
                        prev_pos = 0
                    else:
                        prev_pos = trading_df.loc[i-1, f'Predicted_results_{time_step}']

                    if not initial_position_set:
                        if curr_pos == 0:
                            action = "No action"
                        else:
                            action = "Buy"
                            initial_position_set = True
                    else:
                        last_action = trading_df.loc[i-1, f'action_{time_step}']

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
                    
                    trading_df.loc[i, f'action_{time_step}'] = action
                    
                if not os.path.exists(f'./Backtesting/trading_signal/YOY_{year}/LSTM/'):
                    os.makedirs(f'./Backtesting/trading_signal/YOY_{year}/LSTM/')
               
                trading_df.to_csv(f'./Backtesting/trading_signal/YOY_{year}/LSTM/{ticker}_{day}_signal.csv', index=True)