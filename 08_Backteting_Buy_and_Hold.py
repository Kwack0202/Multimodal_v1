from common_imports import *

tickers = ['IXIC', 'US500','KS11', 'KQ11']
days = ['1day', '5day']

for ticker in tickers:
    stock_data = pd.read_csv(f'./Backtesting/trading_signal/full_period/LSTM/{ticker}_1day_signal.csv')
    # 첫 행과 마지막 행의 Open과 Close 값을 가져옴
    start_price = stock_data.loc[0, 'Open']
    end_price = stock_data.loc[len(stock_data) - 1, 'Close']

    # 전체 기간 수익률 계산
    profit_rate = (end_price - start_price) / start_price * 100

    # Date 열을 날짜 형식으로 변환
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])

    # Date 열을 연도로 추출
    stock_data['Year'] = stock_data['Date'].dt.year

    # 2020년, 2021년, 2022년 데이터 추출
    data_2020 = stock_data[stock_data['Year'] == 2020]
    data_2021 = stock_data[stock_data['Year'] == 2021]
    data_2022 = stock_data[stock_data['Year'] == 2022]

    # 각 연도의 첫 행과 마지막 행의 Open 및 Close 값을 가져옴
    start_price_2020 = data_2020.iloc[0]['Open']
    end_price_2020 = data_2020.iloc[-1]['Close']

    start_price_2021 = data_2021.iloc[0]['Open']
    end_price_2021 = data_2021.iloc[-1]['Close']

    start_price_2022 = data_2022.iloc[0]['Open']
    end_price_2022 = data_2022.iloc[-1]['Close']

    # 각 연도의 수익률 계산
    profit_rate_2020 = (end_price_2020 - start_price_2020) / start_price_2020 * 100
    profit_rate_2021 = (end_price_2021 - start_price_2021) / start_price_2021 * 100
    profit_rate_2022 = (end_price_2022 - start_price_2022) / start_price_2022 * 100

    # 결과를 데이터 프레임으로 만듦
    result = pd.DataFrame({'Year': ['2020~2022', 2020, 2021, 2022],
                        'Start Price': [start_price, start_price_2020, start_price_2021, start_price_2022],
                        'End Price': [end_price, end_price_2020, end_price_2021, end_price_2022],
                        'Profit Price': [(end_price - start_price), (end_price_2020 - start_price_2020), (end_price_2021 - start_price_2021), (end_price_2022 - start_price_2022)],
                        'Profit Rate(%)': [profit_rate, profit_rate_2020, profit_rate_2021, profit_rate_2022]})
        
    if not os.path.exists(f'./Backtesting/trading_results/Buy&Hold/'):
            os.makedirs(f'./Backtesting/trading_results/Buy&Hold/')
                
    result.to_csv(f'./Backtesting/trading_results/Buy&Hold/{ticker}_signal.csv', index=True)