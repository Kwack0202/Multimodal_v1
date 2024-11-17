from common_imports import *

def calculate_indicators(df):
    df['MOM'] = talib.MOM(df["Close"], timeperiod=10)
    df['DMI'] = talib.DX(df["High"], df["Low"], df["Close"], timeperiod=14)
    df['ROC'] = talib.ROC(df["Close"], timeperiod=10)
    df['ADX'] = talib.ADX(df["High"], df["Low"], df["Close"], timeperiod=14)
    df['AROON_OSC'] = talib.AROONOSC(df["High"], df["Low"], timeperiod=14)
    df['CCI'] = talib.CCI(df["High"], df["Low"], df["Close"], timeperiod=9)
    df['RSI'] = talib.RSI(df["Close"], timeperiod=14)
    df['slowk'], df['slowd'] = talib.STOCH(df["High"], df["Low"], df["Close"], fastk_period=12, slowk_period=5, slowk_matype=0, slowd_period=5, slowd_matype=0)
    df['fastk'], df['fastd'] = talib.STOCHF(df["High"], df["Low"], df["Close"], fastk_period=12, fastd_period=5, fastd_matype=0)
    df['CMO'] = talib.CMO(df["Close"], timeperiod=9)
    df['PPO'] = talib.PPO(df["Close"], fastperiod=10, slowperiod=20, matype=0)
    df['UO'] = talib.ULTOSC(df["High"], df["Low"], df["Close"], timeperiod1=7, timeperiod2=14, timeperiod3=28)
    
    return df

def add_labels(df, num_days = 5):
    # 하루 뒤 단일추세
    df['Signal_origin'] = np.where((df['Close'].shift(-1) - df['Close']) / df['Close'] >= 0.00, 1, 0)
    
    # 5일간의 연속적추세
    df['Signal_trend'] = np.where(df['Close'].rolling(window=num_days).mean().shift(-num_days) > df['Close'], 1, 0)
    return df

def process_stock_data(stock_code):
    file_path = f"./csv/origin_data/{stock_code}.csv"
    stock_data = pd.read_csv(file_path)

    # TA 
    stock_data = calculate_indicators(stock_data)
    
    # Data Labelling
    stock_data = add_labels(stock_data, num_days = 5)

    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    mask = (stock_data['Date'].dt.year == 2011)
    filtered_stock_data = stock_data[mask]
    
    if not filtered_stock_data.empty:
        idx = filtered_stock_data.index[0] - 119
        stock_data = stock_data.iloc[idx:, :]

    stock_data = stock_data[stock_data['Date'].dt.year < 2023]
    stock_data = stock_data.reset_index(drop=True)

    stock_data.to_csv(f"{output_dir}{stock_code}.csv", encoding='utf-8', index=False)


# Define the list of stock codes
stock_codes = ['IXIC', 'US500', 'KS11', 'KQ11']

output_dir = './csv/TA_csv/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop through each stock code
for stock_code in tqdm(stock_codes):
    process_stock_data(stock_code)