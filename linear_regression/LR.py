import os, csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df=pd.read_csv('./../data/raw_data/india_data.csv', sep=',',header=None)
# df = df.T
df = np.asarray(df)



print(df.shape)
df_new = df[0].reshape(-1, 1)
LR = LinearRegression()
LR.fit(df_new,df[1])
y_prediction =  LR.predict(df_new)
y_prediction

figure, axis = plt.subplots(1, 1, figsize = (10,10))
axis.plot(np.arange(25), y_prediction)
axis.scatter(np.arange(25), df[1],marker='s')
axis.set_title("actual vs predicted (test set)")
axis.legend(['predicted', 'actual'], loc='upper left')
plt.show()
# df.plot
# pyplot.show()

# plt.plot(df[0], df[1], linestyle='solid', marker='o', label='Number of people using mobile phones in Inida')
# plt.title('Number of Mobile Users in India Vs Year')
# plt.xlabel('Year')
# plt.ylabel('Number of Mobile Users in India')
# plt.savefig('./../plots/MobileUsersVsYear.png')
# plt.show()