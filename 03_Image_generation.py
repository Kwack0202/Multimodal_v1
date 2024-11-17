from common_imports import *

# 이미지 생성용 함수
def plot_candles(pricing, title=None, trend_line = False, volume_bars=False, color_function=None, technicals=None):
    
    def default_color(index, open_price, close_price, low, high):
        return 'b' if open_price[index] > close_price[index] else 'r'
    
    color_function = color_function or default_color
    technicals = technicals or []
    open_price = pricing['Open']
    close_price = pricing['Close']
    low = pricing['Low']
    high = pricing['High']
    oc_min = pd.concat([open_price, close_price], axis=1).min(axis=1)
    oc_max = pd.concat([open_price, close_price], axis=1).max(axis=1)
    
    def plot_trendline(ax, pricing, linewidth=5):
        x = np.arange(len(pricing))
        y = pricing.values
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax.plot(x, p(x), 'g--', linewidth = linewidth)
    
    if volume_bars:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3,1]},figsize=(20,10))
    else:
        fig, ax1 = plt.subplots(1, 1)
    if title:
        ax1.set_title(title)
    fig.tight_layout()
    x = np.arange(len(pricing))
    candle_colors = [color_function(i, open_price, close_price, low, high) for i in x]
    candles = ax1.bar(x, oc_max-oc_min, bottom=oc_min, color=candle_colors, linewidth=0)
    lines = ax1.vlines(x , low, high, color=candle_colors, linewidth=1)
    
    # 추세선 생성
    if trend_line:
        plot_trendline(ax1, pricing['Close'])
    
    ax1.xaxis.grid(True)
    ax1.yaxis.grid(True)
    ax1.xaxis.set_tick_params(which='major', length=3.0, direction='in', top='off')
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)
    ax1.axis(False)

    for indicator in technicals:
        ax1.plot(x, indicator)
    
    if volume_bars:
        volume = pricing['Volume']
        volume_scale = None
        scaled_volume = volume
        if volume.max() > 1000000:
            volume_scale = 'M'
            scaled_volume = volume / 1000000
        elif volume.max() > 1000:
            volume_scale = 'K'
            scaled_volume = volume / 1000
        ax2.bar(x, scaled_volume, color=candle_colors)
        volume_title = 'Volume'
        if volume_scale:
            volume_title = 'Volume (%s)' % volume_scale
        #ax2.set_title(volume_title)
        ax2.xaxis.grid(True)
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])
        ax2.set_yticklabels([])
        ax2.set_xticklabels([])
        ax2.axis(False)
    return fig


def save_candlestick_images(stock_data, seq_lens, route_new):
    for seq_len in seq_lens:
        for i in tqdm(range(0, len(stock_data) - 120 + 1)):
            if seq_len == 120:
                candlestick_data = stock_data.iloc[i:i + seq_len]
            else:
                candlestick_data = stock_data.iloc[i + 120 - seq_len:i + 120]
            candlestick_data = candlestick_data.reset_index(drop=True)

            seq_path = os.path.join(route_new, str(seq_len))
            os.makedirs(seq_path, exist_ok=True)

            fig = plot_candles(candlestick_data, trend_line=False, volume_bars=False)
            fig.savefig(os.path.join(seq_path, f'{i}.png'), dpi=150)
            plt.close(fig)


# 경고 메시지 숨기기
plt.rcParams['figure.max_open_warning'] = 0

# 이미지 생성용 파라미터 =================================================
seq_lens = [120, 60, 20, 5]  # 차트 이미지에 포함되는 시계열 인덱스 리스트
window_len = 1  # 차트 이미지 윈도우 이동 단위

# Main processing loop
stock_codes = ['IXIC', 'US500', 'KS11', 'KQ11']

for stock_code in tqdm(stock_codes):
    stock_data = pd.read_csv(f"./csv/origin_data/{stock_code}.csv")
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data = stock_data[stock_data['Date'] <= '2023']

    mask = (stock_data['Date'].dt.year == 2011)
    filtered_stock_data = stock_data[mask]

    if not filtered_stock_data.empty:
        idx = filtered_stock_data.index[0] - 119
        stock_data = stock_data.iloc[idx:].reset_index(drop=True)

    route_new = os.path.join("./candle_img", stock_code)
    print(f"\n캔들스틱 차트 이미지 생성 : [ {stock_code} ]")

    save_candlestick_images(stock_data, seq_lens, route_new)