from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

data_root_path = r"/home/ajay/my_projects/neural_networks/Part+3+-+" \
                 r"Recurrent+Neural+Networks/Part 3 - Recurrent Neural Networks"
path_ = lambda x: join(data_root_path, x)
# Running Predictions
test_data = pd.read_csv(path_('Google_Stock_Price_Test.csv'))
train_data = pd.read_csv(path_("Google_Stock_Price_Train.csv"))
total_dataset = pd.concat([train_data, test_data], axis=0)[['Open']]
index_start_test = len(total_dataset) - len(test_data)
inputs = total_dataset[index_start_test-60:].values
print(np.shape(inputs))

train_array = train_data[['Open']].values
# test_data = test_data[['Open']].values
sc = MinMaxScaler()
sc.fit(train_array)
inputs_scaled = sc.transform(inputs)


x_test = []
for i in range(60, 80):
    x_test.append(inputs_scaled[i-60:i, 0])
x = np.asarray(x_test)
m, n = np.shape(x)
# Reshaping
x_test = np.reshape(x, [m, n, 1])

model = tf.keras.models.load_model('stock_price.h5')
pred = model.predict(x_test)
predicted_values = sc.inverse_transform(pred)
print('Output Shape', np.shape(predicted_values))
print(predicted_values)
fig, ax = plt.subplots()
ax.plot(test_data[['Open']].values, label='Actual', color='b')
ax.plot(predicted_values, label='Predicted', color='k', linestyle='--')
ax.legend()
plt.show()
