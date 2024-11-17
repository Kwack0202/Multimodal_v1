from common_imports import *

years = ['2020', '2021', '2022']

tickers = ['IXIC', 'US500', 'KS11', 'KQ11']
days = ['1day', '5day']
time_steps = ['5', '20', '60', '120']

votings = ['soft', 'hard']
summary_data = []

for year in years:
    for ticker in tickers:
        for day in days:
            for model in ['LSTM', 'VIT']:
                for time_step in time_steps:
                    # 데이터 프레임 Load
                    df = pd.read_csv(f'./Backtesting/trading_results/YOY_{year}/{model}/{ticker}_{day}_{time_step}_signal.csv', index_col=0)
                    df.columns.values[5:7] = ["Predicted_results", "action"]
                    
                    # ============================================================================================
                    # 종목마다 매수 이후 변동값 계산
                    
                    # Buy와 sell 신호를 기반으로 새로운 컬럼 초기화
                    df['Unrealized_profit'] = 0

                    buy_price = 0  # Buy 시점의 close 가격
                    holding = False  # Buy 후 Holding 중 여부 확인

                    # 데이터프레임 순회
                    for index, row in df.iterrows():
                        if row['action'] == 'Buy':
                            buy_price = row['Close']
                            holding = True
                        elif row['action'] == 'Holding' and holding:
                            # Buy 후 Holding 중인 동안, close 가격에 따라 Unrealized_profit 업데이트
                            df.at[index, 'Unrealized_profit'] = row['Close'] - buy_price
                        elif row['action'] == 'sell' and holding:
                            # Sell 시점에서 Unrealized_profit 업데이트
                            df.at[index, 'Unrealized_profit'] = row['Close'] - buy_price
                            buy_price = 0
                            holding = False
                        elif row['action'] == 'No action' and holding:
                            # Holding 중이지만 Buy 신호 이후 No action 인 경우, Unrealized_profit은 그대로 0 유지
                            pass
                    
                    # 'action' 컬럼이 'sell'인 경우에만 Cumulative_Profit를 0으로 설정하면서 계산
                    df['New_Column'] = df['Cumulative_Profit']
                
                    first_sell_index = df[df['action'] == 'sell'].index[0]
                    df.at[first_sell_index, 'New_Column'] = 0
                    
                    # 'action' 열 값이 'sell'인 행을 찾아서 대체
                    mask = df['action'] == 'sell'
                    df.loc[mask, 'New_Column'] = df['New_Column'].shift(1)
                    df.loc[mask, 'Cumulative_Profit'] = df['Cumulative_Profit'].shift(1)

                    df['Unrealized_profit_plus_Cumulative_Profit'] = df['New_Column'] + df['Unrealized_profit']
                    
                    # 컬럼 제거
                    df = df.drop('New_Column', axis=1) 

                    # ============================================================================================
                    # Holding 기간을 나타내는 컬럼 생성 
                    df['Holding_Period'] = df.groupby((df['action'] != 'Holding').cumsum()).cumcount()
                    
                    # ============================================================================================
                    # 수익률 평가 계산  
                    # 'Buy' 신호가 있는지 확인하고, 없는 경우 초기 투자금액을 0으로 설정
                    if 'Buy' in df['action'].values:
                        initial_investment = df[df['action'] == 'Buy']['Close'].iloc[0]
                    else:
                        initial_investment = 0

                    # 누적 수익률 컬럼 추가
                    df['Cumulative_Return'] = (df['Cumulative_Profit'] / initial_investment) * 100
                        
                    # 누적 미실현 수익률 컬럼 추가
                    df['Unrealized_Return'] = (df['Unrealized_profit'] / initial_investment) * 100
                        
                    # 누적 수익 + 미실현 수익률 컬럼 추가
                    df['Unrealized_profit_plus_Cumulative_Profit_Return'] = (df['Unrealized_profit_plus_Cumulative_Profit'] / initial_investment) * 100
                    
                    # ============================================================================================
                    # 포트폴리오 평가 계산
                    df['Cum_portfolio'] = df['Cumulative_Profit'] + initial_investment
                    
                    df['Unreal_Cum_portfolio'] = df['Unrealized_profit_plus_Cumulative_Profit'] + initial_investment

                    if not os.path.exists(f'./Backtesting/Final_results/YOY_{year}/'):
                            os.makedirs(f'./Backtesting/Final_results/YOY_{year}/')
                            
                    df.to_csv(f"./Backtesting/Final_results/YOY_{year}/{ticker}_{model}_{day}_{time_step}.csv")
                    
                    # ============================================================================================
                    # 매매 결과 총 정리 
                    df['action'] = df['action'].replace('No action', 0)
                    df['action'] = df['action'].replace('Buy', 1)
                    df['action'] = df['action'].replace('sell', -1)
                    
                    # 거래 횟수
                    no_trade = len(df[df['Sell_Signal_Profit'] > 0]) + len(df[df['Sell_Signal_Profit'] < 0])
                    
                    # 승률
                    winning_ratio = len(df[(df['action'] == -1) & (df['Sell_Signal_Profit'] > 0)]) / no_trade if no_trade > 0 else 0
                    
                    # 수익 평균, 손실 평균
                    profit_average = df[df['Sell_Signal_Profit'] > 0]['Sell_Signal_Profit'].mean()
                    loss_average = df[df['Sell_Signal_Profit'] < 0]['Sell_Signal_Profit'].mean()
                    
                    # payoff_ratio, profit_factor
                    payoff_ratio = profit_average / -loss_average if loss_average < 0 else 0
                    profit_factor = -df[df['Sell_Signal_Profit'] > 0]['Sell_Signal_Profit'].sum() / df[df['Sell_Signal_Profit'] < 0]['Sell_Signal_Profit'].sum()
                    
                    # Maximum Drawdown (MDD)
                    peak = df[df['Cum_portfolio'] != 0]['Cum_portfolio'].expanding().max()
                    drawdown = df['Cum_portfolio'] / peak - 1
                    max_drawdown = drawdown.min() * 100
                    
                    peak = df[df['Unreal_Cum_portfolio'] != 0]['Unreal_Cum_portfolio'].expanding().max()
                    drawdown = df['Unreal_Cum_portfolio'] / peak - 1
                    max_portfolio_drawdown = drawdown.min() * 100
                    
                    # 가장 긴 Holding 기간의 값을 찾기
                    max_holding_period = df[df['action'] == 'Holding']['Holding_Period'].max()
                    
                    # 평균 Holding 기간의 값을 찾기
                    mean_holding_period = df[df['action'] == 'Holding']['Holding_Period'].mean()

                    # Maximum profit and maximum loss (최대 실현 수익금액, 손실금액 및 비율)
                    max_profit = df[df['Sell_Signal_Profit'] != 0]['Sell_Signal_Profit'].max()
                    max_profit_return = df[df['Sell_Signal_Return'] != 0]['Sell_Signal_Return'].max()
                    
                    max_loss = df[df['Sell_Signal_Profit'] != 0]['Sell_Signal_Profit'].min()
                    max_loss_return = df[df['Sell_Signal_Return'] != 0]['Sell_Signal_Return'].min()
                    
                    # Maximum profit and maximum loss Ratio (최대 누적수익금액, 손실금액 및 비율 + 최종 수익 비율)
                    max_cum_profit = df[df['Cumulative_Profit'] != 0]['Cumulative_Profit'].max()
                    max_cum_profit_return = df[df['Cumulative_Return'] != 0]['Cumulative_Return'].max()
                    
                    max_cum_loss = df[df['Cumulative_Profit'] != 0]['Cumulative_Profit'].min()
                    max_cum_loss_return = df[df['Cumulative_Return'] != 0]['Cumulative_Return'].min()
                    
                    last_cumulative = df['Cumulative_Profit'].iloc[-1]
                    last_cumulative_return = df['Cumulative_Return'].iloc[-1]
                    
                    # Maximum Unrealized profit and maximum loss (최대 잔고평가 수익금액, 손실금액 및 비율)
                    max_unrealized_profit = df[df['Unrealized_profit_plus_Cumulative_Profit'] != 0]['Unrealized_profit_plus_Cumulative_Profit'].max()
                    max_unrealized_profit_return = df[df['Unrealized_profit_plus_Cumulative_Profit_Return'] != 0]['Unrealized_profit_plus_Cumulative_Profit_Return'].max()
                    
                    max_unrealized_loss = df[df['Unrealized_profit_plus_Cumulative_Profit'] != 0]['Unrealized_profit_plus_Cumulative_Profit'].min()
                    max_unrealized_loss_return = df[df['Unrealized_profit_plus_Cumulative_Profit_Return'] != 0]['Unrealized_profit_plus_Cumulative_Profit_Return'].min()
                    
                    data_frame_name = f"{ticker}_{model}_{day}_{time_step}"

                    summary_data.append([data_frame_name, no_trade, winning_ratio, profit_average, loss_average, payoff_ratio, profit_factor, 
                                        max_drawdown, max_portfolio_drawdown, max_holding_period, mean_holding_period, 
                                        max_profit, max_profit_return, max_loss, max_loss_return,
                                        max_cum_profit, max_cum_profit_return, max_cum_loss, max_cum_loss_return, last_cumulative, last_cumulative_return,
                                        max_unrealized_profit, max_unrealized_profit_return, max_unrealized_loss, max_unrealized_loss_return])
            
            for model in ['Fusion']:
                
                for voting in votings:
                    
                    for ml in ['lgb', 'SVM', 'xgb']:
                        
                        # 데이터 프레임 Load
                        df = pd.read_csv(f'./Backtesting/trading_results/YOY_{year}/{model}/{ticker}_{voting}_{day}_{ml}_signal.csv', index_col=0)
                        df.columns.values[5:7] = ["Predicted_results", "action"]
                        
                        # ============================================================================================
                        # 종목마다 매수 이후 변동값 계산
                        
                        # Buy와 sell 신호를 기반으로 새로운 컬럼 초기화
                        df['Unrealized_profit'] = 0

                        buy_price = 0  # Buy 시점의 close 가격
                        holding = False  # Buy 후 Holding 중 여부 확인

                        # 데이터프레임 순회
                        for index, row in df.iterrows():
                            if row['action'] == 'Buy':
                                buy_price = row['Close']
                                holding = True
                            elif row['action'] == 'Holding' and holding:
                                # Buy 후 Holding 중인 동안, close 가격에 따라 Unrealized_profit 업데이트
                                df.at[index, 'Unrealized_profit'] = row['Close'] - buy_price
                            elif row['action'] == 'sell' and holding:
                                # Sell 시점에서 Unrealized_profit 업데이트
                                df.at[index, 'Unrealized_profit'] = row['Close'] - buy_price
                                buy_price = 0
                                holding = False
                            elif row['action'] == 'No action' and holding:
                                # Holding 중이지만 Buy 신호 이후 No action 인 경우, Unrealized_profit은 그대로 0 유지
                                pass
                        
                        # 'action' 컬럼이 'sell'인 경우에만 Cumulative_Profit를 0으로 설정하면서 계산
                        df['New_Column'] = df['Cumulative_Profit']
                    
                        first_sell_index = df[df['action'] == 'sell'].index[0]
                        df.at[first_sell_index, 'New_Column'] = 0
                    
                        # 'action' 열 값이 'sell'인 행을 찾아서 대체
                        mask = df['action'] == 'sell'
                        df.loc[mask, 'New_Column'] = df['New_Column'].shift(1)
                        df.loc[mask, 'Cumulative_Profit'] = df['Cumulative_Profit'].shift(1)

                        df['Unrealized_profit_plus_Cumulative_Profit'] = df['New_Column'] + df['Unrealized_profit']
                        
                        # 컬럼 제거
                        df = df.drop('New_Column', axis=1) 
                    
                        # ============================================================================================
                        # Holding 기간을 나타내는 컬럼 생성 
                        df['Holding_Period'] = df.groupby((df['action'] != 'Holding').cumsum()).cumcount()
                        
                        # ============================================================================================
                        # 수익률 평가 계산  
                        # 'Buy' 신호가 있는지 확인하고, 없는 경우 초기 투자금액을 0으로 설정
                        if 'Buy' in df['action'].values:
                            initial_investment = df[df['action'] == 'Buy']['Close'].iloc[0]
                        else:
                            initial_investment = 0

                        # 누적 수익률 컬럼 추가
                        df['Cumulative_Return'] = (df['Cumulative_Profit'] / initial_investment) * 100
                        
                        # 누적 미실현 수익률 컬럼 추가
                        df['Unrealized_Return'] = (df['Unrealized_profit'] / initial_investment) * 100
                        
                        # 누적 수익 + 미실현 수익률 컬럼 추가
                        df['Unrealized_profit_plus_Cumulative_Profit_Return'] = (df['Unrealized_profit_plus_Cumulative_Profit'] / initial_investment) * 100
                        
                        # ============================================================================================
                        # 포트폴리오 평가 계산
                        df['Cum_portfolio'] = df['Cumulative_Profit'] + initial_investment
                        
                        df['Unreal_Cum_portfolio'] = df['Unrealized_profit_plus_Cumulative_Profit'] + initial_investment
                        
                        if not os.path.exists(f'./Backtesting/Final_results/YOY_{year}/'):
                            os.makedirs(f'./Backtesting/Final_results/YOY_{year}/')
                            
                        df.to_csv(f"./Backtesting/Final_results/YOY_{year}/{ticker}_{model}_{voting}_{day}_{ml}.csv")
                        
                        # ============================================================================================
                        # 매매 결과 총 정리
                        df['action'] = df['action'].replace('No action', 0)
                        df['action'] = df['action'].replace('Buy', 1)
                        df['action'] = df['action'].replace('sell', -1)
                        
                        # 거래 횟수
                        no_trade = len(df[df['Sell_Signal_Profit'] > 0]) + len(df[df['Sell_Signal_Profit'] < 0])
                        
                        # 승률
                        winning_ratio = len(df[(df['action'] == -1) & (df['Sell_Signal_Profit'] > 0)]) / no_trade if no_trade > 0 else 0
                        
                        # 수익 평균, 손실 평균
                        profit_average = df[df['Sell_Signal_Profit'] > 0]['Sell_Signal_Profit'].mean()
                        loss_average = df[df['Sell_Signal_Profit'] < 0]['Sell_Signal_Profit'].mean()
                        
                        # payoff_ratio, profit_factor
                        payoff_ratio = profit_average / -loss_average if loss_average < 0 else 0
                        profit_factor = -df[df['Sell_Signal_Profit'] > 0]['Sell_Signal_Profit'].sum() / df[df['Sell_Signal_Profit'] < 0]['Sell_Signal_Profit'].sum()
                        
                        # Maximum Drawdown (MDD)
                        peak = df[df['Cum_portfolio'] != 0]['Cum_portfolio'].expanding().max()
                        drawdown = df['Cum_portfolio'] / peak - 1
                        max_drawdown = drawdown.min() * 100
                        
                        peak = df[df['Unreal_Cum_portfolio'] != 0]['Unreal_Cum_portfolio'].expanding().max()
                        drawdown = df['Unreal_Cum_portfolio'] / peak - 1
                        max_portfolio_drawdown = drawdown.min() * 100
                            
                        # 가장 긴 Holding 기간의 값을 찾기
                        max_holding_period = df[df['action'] == 'Holding']['Holding_Period'].max()
                        
                        # 평균 Holding 기간의 값을 찾기
                        mean_holding_period = df[df['action'] == 'Holding']['Holding_Period'].mean()

                        # Maximum profit and maximum loss (최대 실현 수익금액, 손실금액 및 비율)
                        max_profit = df[df['Sell_Signal_Profit'] != 0]['Sell_Signal_Profit'].max()
                        max_profit_return = df[df['Sell_Signal_Return'] != 0]['Sell_Signal_Return'].max()
                        
                        max_loss = df[df['Sell_Signal_Profit'] != 0]['Sell_Signal_Profit'].min()
                        max_loss_return = df[df['Sell_Signal_Return'] != 0]['Sell_Signal_Return'].min()
                        
                        # Maximum profit and maximum loss Ratio (최대 누적수익금액, 손실금액 및 비율 + 최종 수익 비율)
                        max_cum_profit = df[df['Cumulative_Profit'] != 0]['Cumulative_Profit'].max()
                        max_cum_profit_return = df[df['Cumulative_Return'] != 0]['Cumulative_Return'].max()
                        
                        max_cum_loss = df[df['Cumulative_Profit'] != 0]['Cumulative_Profit'].min()
                        max_cum_loss_return = df[df['Cumulative_Return'] != 0]['Cumulative_Return'].min()
                        
                        last_cumulative = df['Cumulative_Profit'].iloc[-1]
                        last_cumulative_return = df['Cumulative_Return'].iloc[-1]
                        
                        # Maximum Unrealized profit and maximum loss (최대 잔고평가 수익금액, 손실금액 및 비율)
                        max_unrealized_profit = df[df['Unrealized_profit_plus_Cumulative_Profit'] != 0]['Unrealized_profit_plus_Cumulative_Profit'].max()
                        max_unrealized_profit_return = df[df['Unrealized_profit_plus_Cumulative_Profit_Return'] != 0]['Unrealized_profit_plus_Cumulative_Profit_Return'].max()
                        
                        max_unrealized_loss = df[df['Unrealized_profit_plus_Cumulative_Profit'] != 0]['Unrealized_profit_plus_Cumulative_Profit'].min()
                        max_unrealized_loss_return = df[df['Unrealized_profit_plus_Cumulative_Profit_Return'] != 0]['Unrealized_profit_plus_Cumulative_Profit_Return'].min()
                            
                        data_frame_name = f"{ticker}_{model}_{voting}_{day}_{ml}"

                        summary_data.append([data_frame_name, no_trade, winning_ratio, profit_average, loss_average, payoff_ratio, profit_factor,
                                            max_drawdown, max_portfolio_drawdown, max_holding_period, mean_holding_period, 
                                            max_profit, max_profit_return, max_loss, max_loss_return,
                                            max_cum_profit, max_cum_profit_return, max_cum_loss, max_cum_loss_return, last_cumulative, last_cumulative_return,
                                            max_unrealized_profit, max_unrealized_profit_return, max_unrealized_loss, max_unrealized_loss_return])
        
        summary_df = pd.DataFrame(summary_data, columns=['Data_Frame_Name', 'No_Trade', 'Winning_Ratio', 'Profit_Average', 'Loss_Average', 'Payoff_Ratio', 'Profit_Factor',
                                                        'MDD', 'MDD(portfolio)', 'Max_Holding_Period', 'Mean_Holding_Period', 
                                                        'Max_Profit', 'Max_Profit_Return', 'Max_Loss', 'Max_Loss_Return',
                                                        'Max_Cum_Profit', 'Max_Cum_Profit_Return', 'Max_Cum_Loss', 'Max_Cum_Loss_Return', 'Last_Cumulative', 'Last_Cumulative_Return',
                                                        'Max_Portfolio_Profit', 'Max_Portfolio_Profit_Return', 'Max_Portfolio_Loss', 'Max_Portfolio_Loss_Return',])
        summary_df.to_csv(f'Final_results_Summary_YOY_{year}.csv')