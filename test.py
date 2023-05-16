import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

data = pd.read_excel(io = r'/home/llc/Projects/bishe/totalFeatures.xls')
nrow = len(data)
ncol = len(data.columns)
row_list = []
col_list = []
for i in range(nrow):
    row = data.iloc[i, :ncol - 1]
    col = data.iloc[i, ncol - 1:]
    row_list.append(row.values)
    col_list.append(col.values[0])
X = np.array(row_list)
y = np.array(col_list)

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

