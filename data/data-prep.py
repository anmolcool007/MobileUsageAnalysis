import os, csv
import pandas as pd
import numpy as np

df=pd.read_csv('./raw_data/india_data.csv', sep=',',header=None)
df = np.asarray(df)
df = df.astype(np.float64)

print(df)