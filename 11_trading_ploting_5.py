from common_imports import *

import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# 파일 경로 설정
file_path = './Backtesting/Final_results/full_period/*.csv'

# CSV 파일 리스트 불러오기
csv_files = glob.glob(file_path)

# 모든 CSV 파일을 대상으로 시각화 저장
for file in csv_files:
    # CSV 파일 불러오기
    df = pd.read_csv(file)
    
    # 날짜를 Datetime 형식으로 변환
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 시각화
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Y축 0에 선 추가
    ax.axhline(y=0, color='gray', linestyle='--')
    
    # Sell_Signal_Return 값에 비례한 원 그리기
    marker_size = 30 * abs(df['Sell_Signal_Return'])
    colors = ['red' if x >= 0 else 'blue' for x in df['Sell_Signal_Return']]
    ax.scatter(df['Date'], df['Sell_Signal_Return'], s=marker_size, alpha=0.5, color=colors, label='Sell Signal Return')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Sell_Signal_Return')
    ax.set_title('Sell Signal Return Visualization')
    ax.legend()
    
    plt.xticks(rotation=45)
    
    # 파일 이름에서 확장자 제거하고 저장
    file_name = os.path.splitext(os.path.basename(file))[0]
    
    output_directory = './Backtesting/trading_plot/return_visual/'
    os.makedirs(output_directory, exist_ok=True)
    
    plt.savefig(f'./Backtesting/trading_plot/return_visual/{file_name}_visualization.png')
    plt.close()