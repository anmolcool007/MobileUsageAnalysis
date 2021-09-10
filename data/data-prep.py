import os, csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import yticks

df=pd.read_csv('./raw_data/india_data.csv', sep=',',header=None)
# df = df.T
df = np.asarray(df)
df = df.astype(np.float64)
# df.plot
# pyplot.show()

plt.plot(df[0], df[1], linestyle='solid', marker='o', label='Number of people using mobile phones in Inida')
plt.title('Number of Mobile Users in India Vs Year')
plt.xlabel('Year')
plt.ylabel('Number of Mobile Users in India')
plt.savefig('./../plots/MobileUsersVsYear.png')
plt.show()