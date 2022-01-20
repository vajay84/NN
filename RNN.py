import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
import matplotlib.pyplot as plt
import joblib
import pandas as pd
from os.path import join
from sklearn.preprocessing import StandardScaler, MinMaxScaler


data_root_path = r"/home/ajay/my_projects/neural_networks/Part+3+-+" \
                 r"Recurrent+Neural+Networks/Part 3 - Recurrent Neural Networks"
path_ = lambda x: join(data_root_path, x)
# Importing Training Set
train = pd.read_csv(path_("Google_Stock_Price_Train.csv"))
X = train[['Open']].values
X_scaled = MinMaxScaler().fit_transform(X)

# Creating dataset with 60 timesteps and 1 output
x_train = []
y_train = []
for i in range(60, len(X)):
    x_train.append(X_scaled[i-60:i, 0])
    y_train.append(X_scaled[i, 0])
x, y = np.asarray(x_train), np.asarray(y_train)
m, n = np.shape(x)
# Reshaping
x_train = np.reshape(x, [m, n, 1])
print(np.shape(x_train))

# RNN
regressor = Sequential()
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(n, 1)))
regressor.add(Dropout(rate=0.2))
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(rate=0.2))
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(rate=0.2))
regressor.add(LSTM(units=50, return_sequences=False))
regressor.add(Dropout(rate=0.2))
regressor.add(Dense(1))

optimizer = tf.keras.optimizers.Adam()
regressor.compile(optimizer=optimizer, loss=tf.keras.losses.mse,
                  metrics=[tf.keras.metrics.MAE])
callbcks = [tf.keras.callbacks.EarlyStopping(patience=10)]
history = regressor.fit(x_train, y, epochs=100, batch_size=32, validation_split=0.2)

