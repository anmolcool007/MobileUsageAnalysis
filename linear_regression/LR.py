import os, csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

df=pd.read_csv('./../data/raw_data/india_data.csv', sep=',',header=None)
df = np.asarray(df)

#calculating Linear Regression
print(df.shape)
df_new = df[0].reshape(-1, 1)
LR = LinearRegression()
LR.fit(df_new,df[1])
y_prediction =  LR.predict(df_new)
y_prediction

#Normalizing for calculating MSE 
scaler_x = preprocessing.MinMaxScaler()
scaler_y = preprocessing.MinMaxScaler()
df_new=df[1].reshape(-1,1)
y_prediction_new = y_prediction.reshape(-1,1)
scaler_x.fit(df_new)
df_new=scaler_x.transform(df_new)
scaler_y.fit(y_prediction_new)
y_prediction_new=scaler_y.transform(y_prediction_new)
#calculating MSE
mse = mean_squared_error(df_new, y_prediction_new)
print("Linear Regression MSE(Normalized) = ")
print(mse)

#Plotting
figure, axis = plt.subplots(1, 1, figsize = (10,10))
axis.plot(np.arange(1995,2020), y_prediction, color='red')
axis.scatter(np.arange(1995,2020), df[1],marker='s')
plt.title('Linear Fit of Number of Mobile Users in India Vs Year')
plt.xlabel('Year')
plt.ylabel('Number of Mobile Users in India')
plt.savefig('./../plots/LinearRegression.png')
plt.show()