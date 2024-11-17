from common_imports import *

# 종목 별 이미지 생성 기간과 동일한 인덱스의 메타 데이터 저장
start_day = '2010-06-01'
end_day = '2023-01-10'

stock_codes = ['IXIC', 'KS11', 'US500', 'KQ11']

if not os.path.exists('./csv/origin_data/'):
    os.makedirs('./csv/origin_data/')
    
for stock_code in tqdm(stock_codes):
    stock_data = pd.DataFrame(fdr.DataReader(stock_code, start_day, end_day))
    stock_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
    stock_data = stock_data.reset_index()
    
    if stock_code in ['KS11', 'KQ11']:
        stock_data = stock_data.dropna()
        stock_data = stock_data.reset_index(drop=True)
        
    stock_data.to_csv(f"./csv/origin_data/{stock_code}.csv", encoding='utf-8', index=False)