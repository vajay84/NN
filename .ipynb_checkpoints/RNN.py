import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import pandas as pd
from os.path import join
from sklearn.preprocessing import StandardScaler, MinMaxScaler


data_root_path = r"/home/ajay/my_projects/neural_networks/Part+3+-+Recurrent" \
                 r"+Neural+Networks/Part 3 - Recurrent Neural Networks"
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
x, y = np.array(x_train), np.array(y_train)
m, n = np.shape(x)
# Reshaping
x_train = np.reshape(x, [m, n, 1])
print(np.shape(x_train))

# RNN

