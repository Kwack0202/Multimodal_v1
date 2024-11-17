from common_imports import *

# Tensorflow
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.client import device_lib

# image processing
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# model / neural network
from tensorflow.keras import Sequential, layers, callbacks, backend
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# =================================================================================
print(tf.__version__)
print(device_lib.list_local_devices())

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 8)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

# =================================================================================
# 데이터 프레임 생성 함수
def create_dataframe(data, name):
    df = pd.DataFrame({'image_name': [f"{i}.png" for i in range(len(data) - 119)]})
    df['Signal_origin'] = data['Signal_origin'][119:].reset_index(drop=True)
    df['Signal_trend'] = data['Signal_trend'][119:].reset_index(drop=True)
    return df

# 이미지 복사 함수
def copy_images(dataframe, split_limit, source_folder, train_dir_origin, test_dir_origin, signal_column):
    # 데이터프레임 Train / Test Split
    train_df = dataframe.iloc[:split_limit]
    test_df = dataframe.iloc[split_limit:]
    
    # Train 데이터 복사
    for index, row in train_df.iterrows():
        image_name = row['image_name']
        signal_value = row[signal_column]

        # 대상 폴더 경로 생성
        destination_subfolder = os.path.join(train_dir_origin, str(signal_value))
        os.makedirs(destination_subfolder, exist_ok=True)

        # 이미지를 대상 폴더로 복사
        source_path = os.path.join(source_folder, image_name)
        destination_path = os.path.join(destination_subfolder, image_name)
        shutil.copy(source_path, destination_path)
    
    # Test 데이터 복사
    for index, row in test_df.iterrows():
        image_name = row['image_name']
        signal_value = row[signal_column]

        # 대상 폴더 경로 생성
        destination_subfolder = os.path.join(test_dir_origin, str(signal_value))
        os.makedirs(destination_subfolder, exist_ok=True)

        # 이미지를 대상 폴더로 복사
        source_path = os.path.join(source_folder, image_name)
        destination_path = os.path.join(destination_subfolder, image_name)
        shutil.copy(source_path, destination_path)
        
# 폴더 복사 함수
def copy_folders(source_dir, target_dir, index, period, signal):
    source_folder = os.path.join(source_dir, index, period, signal)
    target_folder = os.path.join(target_dir, period, signal)

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for filename in os.listdir(source_folder):
        # Rename 파일
        new_filename = f'{index}_{filename}'
        source_file = os.path.join(source_folder, filename)
        target_file = os.path.join(target_folder, new_filename)
        shutil.copy(source_file, target_file)
        
# =================================================================================
        
# CSV 파일 로드
KOSPI = pd.read_csv("./TA_csv/KS11.csv")
KOSDAQ = pd.read_csv("./TA_csv/KQ11.csv")
NASDAQ = pd.read_csv("./TA_csv/IXIC.csv")
SP500 = pd.read_csv("./TA_csv/US500.csv")

# 데이터프레임 생성
KOSPI_img = create_dataframe(KOSPI, "KS11")
KOSDAQ_img = create_dataframe(KOSDAQ, "KQ11")
NASDAQ_img = create_dataframe(NASDAQ, "IXIC")
SP500_img = create_dataframe(SP500, "US500")

# 설정 값
tickers = ['IXIC', 'US500', 'KS11', 'KQ11']
time_steps = ['5', '20', '60', '120']

paths = [
    ('./VIT_data/signal_origin/train/', './VIT_data/signal_origin/train/All_stock/'),
    ('./VIT_data/signal_trend/train/', './VIT_data/signal_trend/train/All_stock/')
]

# =================================================================================

# 이미지 파일이 있는 디렉토리 경로 설정
for ticker in tqdm(tickers):
    if ticker == 'KS11':
        split_limit = -740
        dataframe = KOSPI_img
    elif ticker == 'KQ11':
        split_limit = -740
        dataframe = KOSDAQ_img
    elif ticker == 'IXIC':
        split_limit = -756
        dataframe = NASDAQ_img
    elif ticker == 'US500':
        split_limit = -756
        dataframe = SP500_img
    else:
        continue

    for time_step in time_steps:
        source_folder = f"./candle_img/{ticker}/{time_step}/"
        
        # Signal_Origin 복사
        train_dir_origin = f'./VIT_data/signal_origin/train/{ticker}/{time_step}/'
        test_dir_origin = f'./VIT_data/signal_origin/test/{ticker}/{time_step}/'
        copy_images(dataframe, split_limit, source_folder, train_dir_origin, test_dir_origin, 'Signal_origin')

        # Signal_Trend 복사
        train_dir_trend = f'./VIT_data/signal_trend/train/{ticker}/{time_step}/'
        test_dir_trend = f'./VIT_data/signal_trend/test/{ticker}/{time_step}/'
        copy_images(dataframe, split_limit, source_folder, train_dir_trend, test_dir_trend, 'Signal_trend')

# =================================================================================

# 모든 경로에 대해 폴더 생성 및 파일 복사
for source_path, target_path in paths:
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    for index in tickers:
        for period in time_steps:
            for signal in ['0', '1']:
                copy_folders(source_path, target_path, index, period, signal)

print("이미지 복사 및 이름 변경이 완료되었습니다.")