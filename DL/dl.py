import os, csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import tensorflow as tf
from time import time
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers import LSTM
# from keras.optimizers import adam
from time import time
from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Bidirectional, LSTM
from keras.layers.convolutional import Convolution1D
from keras.layers.core import Dropout, Activation, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import *
from keras.callbacks import EarlyStopping
from sklearn import preprocessing

df=pd.read_csv('./../data/raw_data/india_data.csv', sep=',',header=None)
df = df.T
df = np.asarray(df)


print(df.shape)
x_train, x_test, y_train, y_test = train_test_split(df[:,0],df[:,1], test_size=0.3)
print(x_train, x_test, y_train, y_test)

scaler_x = preprocessing.MinMaxScaler()
scaler_y = preprocessing.MinMaxScaler()
x_train=x_train.reshape(-1,1)
y_train=y_train.reshape(-1,1)
x_test = x_test.reshape(-1,1)
y_test = y_test.reshape(-1,1)

print(scaler_x.fit(x_train))
x_train=scaler_x.transform(x_train)
print(scaler_y.fit(y_train))
y_train=scaler_y.transform(y_train)

np.expand_dims(x_train, axis=-1)
np.expand_dims(x_test, axis=-1)
print(x_train.shape)
model = Sequential() 
model.add(Dense(512,input_dim =1 ))
model.add(LeakyReLU())
model.add(Dense(256,input_dim =1 ))
model.add(LeakyReLU())
model.add(Dropout(0.1))
model.add(Dense(128,input_dim =1 ))
model.add(LeakyReLU())
model.add(Dropout(0.1))
model.add(Dense(64,input_dim =1 ))
model.add(LeakyReLU())
model.add(Dropout(0.1))
model.add(Dense(32,input_dim =1 ))
model.add(LeakyReLU())
model.add(Dropout(0.1))
model.add(Dense(16,input_dim =1 ))
model.add(LeakyReLU())
model.add(Dropout(0.1))
model.add(Dense(8,input_dim =1 ))
model.add(LeakyReLU())
model.add(Dropout(0.1))
model.add(Dense(1))
print(model.summary())
print(x_train)
model.compile(loss = 'mse', optimizer= 'adam')
earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=80,  verbose=1, mode='min')

start = time()
print("start:",0)
history = model.fit(x_train,y_train, epochs = 200, batch_size=2, verbose = 1, shuffle = False, callbacks=[earlystop], validation_split=0.1)
print("end:",time()-start)




loss = history.history

plt.figure(1)
plt.subplot(221)
plt.plot(history.history['loss'])
plt.title('Training Loss')
plt.subplot(222)
plt.plot(history.history['val_loss'])
plt.title('Validation Loss')
plt.tight_layout()
plt.savefig('./../plots/DNNLoss.png', dpi=300)
plt.show()

scaler_x.fit(x_test)
x_test=scaler_x.transform(x_test)
scaler_x.fit(y_test)
y_test=scaler_x.transform(y_test)

y_pred = model.predict(x_test)
print("Normaalized MSE (DNN)",mean_squared_error(y_pred,y_test))

y_pred = scaler_y.inverse_transform(y_pred)
y_test = scaler_y.inverse_transform(y_test)

plt.figure(figsize=(20,10))
plt.plot( y_test, '.-', color='red', label='Real values', alpha=0.5)
plt.plot( y_pred, '.-', color='blue', label='Predicted values', alpha=1)
plt.title('DNN prediction of Number of Mobile Users in India Vs Year', fontsize=18)
plt.xlabel('Data points (random years)', fontsize=18)
plt.ylabel('Number of Mobile Users in India', fontsize=18)
plt.legend(prop={'size':14})
plt.savefig("./../plots/DNN.png")
plt.show()
