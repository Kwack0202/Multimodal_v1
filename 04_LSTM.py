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
# Set random seed

seed_value = 42
np.random.seed(seed_value)
random.seed(seed_value)
tf.random.set_seed(seed_value)

# =================================================================================
# 모델 파라미터 조정 부분
''' datasets = {
    'IXIC': {'split_num': 756, 'layers': [64, 32, 32]},
    'US500': {'split_num': 756, 'layers': [64, 32, 64]},
    'KS11': {'split_num': 740, 'layers': [32, 16]},
    'KQ11': {'split_num': 740, 'layers': [32, 16]}
} '''

datasets = {
    'IXIC': {'split_num': 756, 'layers': [32, 16, 16]},
    'US500': {'split_num': 756, 'layers': [32, 16, 16]},
    'KS11': {'split_num': 740, 'layers': [32, 16]},
    'KQ11': {'split_num': 740, 'layers': [32, 16]}
}

days = ['1day', '5day']
time_steps_list = [5, 20, 60, 120]

epoch = 30
batch_size = 16

# =================================================================================
# 결과 저장 경로 생성 함수
def ensure_directories_exist(base_dirs, sub_dirs):
    for base_dir in base_dirs:
        for sub_dir in sub_dirs:
            path = os.path.join(base_dir, sub_dir)
            os.makedirs(path, exist_ok=True)
            

# 입력 벡터 구성             
def preprocess_data(data, time_steps, target_column):
    features = data[['MOM', 'DMI', 'ROC', 'ADX', 'AROON_OSC', 'CCI', 'RSI', 'slowk', 'slowd', 'fastk', 'fastd', 'CMO', 'PPO', 'UO']]
    target = data[target_column]
    
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    X, Y = [], []
    for i in range(len(features_scaled) - time_steps + 1):
        X.append(features_scaled[i:i + time_steps])
        Y.append(target[i + time_steps - 1])
    
    return np.array(X), np.array(Y).reshape(-1, 1)

# 모델 생성 및 compile
def create_lstm_model(input_shape, layers):
    model = Sequential()
    model.add(LSTM(units=layers[0], input_shape=input_shape, return_sequences=True))
    model.add(Dense(layers[1], activation='relu'))
    
    if len(layers) > 2:
        model.add(Dense(layers[2], activation='relu'))
        
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    
    return model

# 모델 결과 저장
def save_results(train_X, train_Y, test_X, test_Y, model, day, dataset_name, time_steps, threshold):
    for data_type, X, Y in [('train', train_X, train_Y), ('test', test_X, test_Y)]:
        predictions = model.predict(X)
        single_predictions = np.mean(predictions, axis=1)
        predicted_result_df = pd.DataFrame({'True_Label': Y.flatten(), 'Predicted_results': single_predictions.flatten()})
        predicted_result_df.to_csv(f'./csv/model_results/LSTM_classification_single_results_{data_type}/{day}/{dataset_name}_LSTM_{time_steps}.csv', index=True)

        binary_predictions = (single_predictions > threshold).astype(int)
        result_df = pd.DataFrame({'True_Label': Y.flatten(), 'Predicted_Label': binary_predictions.flatten()})
        result_df.to_csv(f'./csv/model_results/LSTM_classification_binary_results_{data_type}/{day}/{dataset_name}_LSTM_{time_steps}_{data_type}_results_mean.csv', index=True)
        
        if data_type == 'test':
            class_report_df = pd.DataFrame(classification_report(result_df['True_Label'], result_df['Predicted_Label'], output_dict=True)).transpose()
            class_report_df.to_csv(f'./csv/model_results/LSTM_classification_report/{day}/{dataset_name}_LSTM_{time_steps}_mean.csv', index=True)

# =================================================================================
base_dirs = [
    './csv/model_results/LSTM_classification_single_results_train',
    './csv/model_results/LSTM_classification_single_results_test',
    './csv/model_results/LSTM_classification_binary_results_train',
    './csv/model_results/LSTM_classification_binary_results_test',
    './csv/model_results/LSTM_classification_report'
]

for day in days:
    ensure_directories_exist(base_dirs, [day])

# =================================================================================
for dataset_name, params in tqdm(datasets.items()):
    data = pd.read_csv(f'./csv/TA_csv/{dataset_name}.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    filtered_stock_data = data[data['Date'].dt.year == 2011]
    
    for time_steps in tqdm(time_steps_list):
        idx = filtered_stock_data.index[0] - time_steps + 1
        data_temp = data.iloc[idx:].reset_index(drop=True)

        for day in days:
            target_column = 'Signal_origin' if day == '1day' else 'Signal_trend'
            
            train_X, train_Y = preprocess_data(data_temp, time_steps, target_column)
            
            test_X, test_Y = train_X[-params['split_num']:], train_Y[-params['split_num']:]
            train_X, train_Y = train_X[:-params['split_num']], train_Y[:-params['split_num']]
            
            model = create_lstm_model((time_steps, train_X.shape[2]), params['layers'])
            model.fit(train_X, train_Y, epochs = epoch, batch_size = batch_size)
            
            if not os.path.exists(f'./saved_model/{dataset_name}/'):
                os.makedirs(f'./saved_model/{dataset_name}/')
                
            model.save(f'./saved_model/{dataset_name}/LSTM_{day}_{time_steps}.h5')

            train_predictions = model.predict(train_X)
            train_threshold = np.mean(train_predictions)

            save_results(train_X, train_Y, test_X, test_Y, model, day, dataset_name, time_steps, train_threshold)