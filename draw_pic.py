# %%
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_excel(io = 'totalFeatures.xls')
nrow = len(data)
ncol = len(data.columns)
row_list = []
col_list = []
for i in range(ncol):
    col = data.iloc[:, i]
    col_list.append(col.values)
col_list = np.array(col_list)
y = col_list[79, :]
X = col_list[:79, :]

# %%
name_list = []
with open('name.txt', "r+", encoding= 'utf-8') as name_data:
    for i in name_data:
        name_list.append(i.split('\n')[0])
name_list.remove('')

# %%
dict = {}
for i in range(len(name_list)):
    dict[name_list[i]] = X[i]

# %%


# %%
def draw_pic(i, j, title):
    for num in range(i, j):
        x_axis = range(0, len(X[0]))
        plt.plot(x_axis, dict[name_list[num]], label = name_list[num])
    plt.title(title)
    plt.show()

# %%
draw_pic(0, 7, "Social App Data")


