# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 13:14:59 2019

@author: agupta466
"""

import os
import pandas as pd
import numpy as np
from math import sqrt
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.optimizers import SGD

path = 'C:\\Users\\agupta466\\OneDrive - DXC Production\\LSTM - Forecasting'

## Importing the complete dataset
dataset = pd.read_csv(str(path)+'\Milho1_SP.csv',header=0,index_col=0)
real_prod = dataset.iloc[84:96,0:1]
dataset+=1
dataset = np.log(dataset)
dataset_train = dataset.iloc[0:84,:]
dataset_test = dataset.iloc[84:96,:]

training_set = dataset_train.iloc[:,0:1].values

## Feature Scaling
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)

## Creating a data structure with 24 timesteps and 1 output
X_train = []
y_train = []

for i in range(36, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-36:i,0])
    y_train.append(training_set_scaled[i,0])

X_train , y_train = np.array(X_train), np.array(y_train)

## Reshaping to feed to LSTM (RNN)
X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))

## Initializing RNN
regressor = Sequential()

## Adding the first LSTM layer and some dropout regularization
regressor.add(LSTM(units = 50, return_sequences=True, input_shape = (X_train.shape[1],1)))
regressor.add(Dropout(0.2))

## Adding the second LSTM layer and some dropout regularization
regressor.add(LSTM(units = 50, return_sequences=True))
regressor.add(Dropout(0.2))

## Adding the thrid LSTM layer and some dropout regularization
regressor.add(LSTM(units = 50, return_sequences=True))
regressor.add(Dropout(0.2))

## Adding the fourth LSTM layer and some dropout regularization
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

## Adding the Output Layer
regressor.add(Dense(units = 1))

## Getting the Predicted Production
dataset_total = pd.concat((dataset_train['Production'], dataset_test['Production']), axis = 0)
inputs = dataset_total[len(dataset_total)-len(dataset_test) - 36:].values
inputs = inputs.reshape(-1,1)
inputs = sc.fit_transform(inputs)

X_test = []
y_test = []

for i in range(36,48):
    X_test.append(inputs[i-36:i,0])
    y_test.append(inputs[i,0])
    
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

## Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['accuracy'])

## Fitting the RNN to the training set
history = regressor.fit(X_train, y_train, epochs = 100, batch_size = 12, verbose = 2, shuffle = False, validation_data = (X_test,y_test))
print(history.history.keys())

## Save the weights
regressor.save_weights(str(path)+'\lstm_weights.h5')

## Plot the training loss v/s validation loss
plt.ylim(0,0.2)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

## Making Predictions and Visualization the Results
## Getting Real Production Data 2019
#real_prod = dataset_test.iloc[:,0:1]

predicted_prod = regressor.predict(X_test)
predicted_prod = sc.inverse_transform(predicted_prod)
predicted_prod = np.exp(predicted_prod)
predicted_prod -= 1

real_prod['Production'] += 1

predicted_prod = pd.DataFrame(predicted_prod, columns=['Production'])
predicted_prod['Production']=predicted_prod['Production'].apply(lambda x: 0 if x <= 10 else x)
predicted_prod+=1

predicted_prod = np.array(predicted_prod)

## Calculate MAPE
mape = np.mean(np.abs((real_prod - predicted_prod) / real_prod)) * 100
print('Test MAPE: %.3f' % mape)

## Calculate RMSE
rmse = sqrt(mean_squared_error(real_prod,predicted_prod))
print('Test RMSE: %.3f' % rmse)

real_prod['Production'] -= 1
predicted_prod -=1

plt.figure(figsize=(10,5))
plt.plot(real_prod, color = 'red', label = 'Real Production')
plt.plot(predicted_prod, color = 'blue', label = 'Predicted Production')
plt.title('Actual v/s Predicted Production')
plt.xlabel('Time')
plt.ylabel('Production')
plt.legend()
plt.show()