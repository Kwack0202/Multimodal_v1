from common_imports import *

# Tensorflow
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.client import device_lib

# image processing
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# model / neural network
from tensorflow.keras import models, Sequential, layers, callbacks, backend
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import tensorflow_addons as tfa

#Vision Transformer
from vit_keras import vit

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
# Set random seed

seed_value = 42
np.random.seed(seed_value)
random.seed(seed_value)
tf.random.set_seed(seed_value)

# =================================================================================
def create_dataframe(data, name):
    df = pd.DataFrame({'image_name': [f"{i}.png" for i in range(len(data) - 119)]})
    df['Signal_origin'] = data['Signal_origin'][119:].reset_index(drop=True)
    df['Signal_trend'] = data['Signal_trend'][119:].reset_index(drop=True)
    return df


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
# =================================================================================
# Function to ensure directories exist
def ensure_directories_exist(base_dirs, sub_dirs):
    for base_dir in base_dirs:
        for sub_dir in sub_dirs:
            path = os.path.join(base_dir, sub_dir)
            os.makedirs(path, exist_ok=True)

# Function to preprocess image data
def preprocess_image_data(data_dir, img_width, img_height, batch_size=32):
    datagen = ImageDataGenerator(rescale=1. / 255)
    generator = datagen.flow_from_directory(
        data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    return generator

# Function to create and compile the Vision Transformer model
def create_vit_model():
    vit_model = vit.vit_b32(
        image_size=224,
        activation='sigmoid',
        pretrained=True,
        include_top=False,
        pretrained_top=False,
        classes=1
    )

    model = models.Sequential()
    model.add(vit_model)
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))

    optimizer = tfa.optimizers.RectifiedAdam(learning_rate=1e-4)

    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.2),
                  metrics=['accuracy'])
    return model

# Function to save results
def save_results(predictions, true_labels, base_dir, filename_prefix):
    single_predictions = np.mean(predictions, axis=1)
    binary_predictions = (single_predictions > single_predictions.mean()).astype(int)

    result_df = pd.DataFrame({'True_Label': true_labels, 'Predicted_Label': binary_predictions.flatten()})
    result_df.to_csv(f'{base_dir}/{filename_prefix}_results.csv', index=True)

    class_report_df = pd.DataFrame(classification_report(true_labels, binary_predictions, output_dict=True)).transpose()
    class_report_df.to_csv(f'{base_dir}/{filename_prefix}_report.csv', index=True)

# Main loop
tickers = ['IXIC', 'US500', 'KS11', 'KQ11']
time_steps = ['5', '20', '60', '120']
days = ['1day', '5day']

base_dirs = [
    './csv/model_results/VIT_classification_report',
    './csv/model_results/VIT_classification_single_results_train',
    './csv/model_results/VIT_classification_single_results_test',
    './csv/model_results/VIT_classification_binary_results_train',
    './csv/model_results/VIT_classification_binary_results_test'
]

for day in days:
    ensure_directories_exist(base_dirs, [day])

for time_step in time_steps:
    for day in days:
        signal = 'signal_origin' if day == '1day' else 'signal_trend'
        signal_col = 'Signal_origin' if day == '1day' else 'Signal_trend'

        # Image data loader
        train_ALL_data_dir = f'./VIT_data/{signal}/train/All_stock/{time_step}/'
        img_width, img_height = 224, 224

        train_all_generator = preprocess_image_data(train_ALL_data_dir, img_width, img_height)

        # Modeling Prepare
        model_vision_transformer = create_vit_model()

        with tf.device('/device:GPU:0'):
            model_vision_transformer.fit(train_all_generator, epochs=30, verbose=1)

        for ticker in tickers:
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

            train_data_dir = f'./VIT_data/{signal}/train/{ticker}/{time_step}/'
            test_data_dir = f'./VIT_data/{signal}/test/{ticker}/{time_step}/'

            # DataFrame Train / Test Split
            train_df = dataframe.iloc[:split_limit]
            test_df = dataframe.iloc[split_limit:]

            train_generator = preprocess_image_data(train_data_dir, img_width, img_height, batch_size=1)
            test_generator = preprocess_image_data(test_data_dir, img_width, img_height, batch_size=1)

            # Model Predict train
            with tf.device('/device:GPU:0'):
                train_predict = model_vision_transformer.predict(train_generator)

            save_results(train_predict, train_df[signal_col], 
                         f'./csv/model_results/VIT_classification_binary_results_train/{day}', 
                         f'{ticker}_VIT_{time_step}_train')

            # Model Predict test
            with tf.device('/device:GPU:0'):
                test_predict = model_vision_transformer.predict(test_generator)

            save_results(test_predict, test_df[signal_col], 
                         f'./csv/model_results/VIT_classification_binary_results_test/{day}', 
                         f'{ticker}_VIT_{time_step}_test')