import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2

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
print(y)
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X = SelectKBest(chi2, k = 1).fit_transform(X, y)
print(X)
x_train,x_test,y_train,y_test = train_test_split(X,y,random_state=14)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

predictor = svm.SVC(gamma='scale', C=1.0, decision_function_shape='ovr', kernel='rbf')
predictor.fit(X, y)

result = predictor.predict(x_test)
total_correct = 0
for i in range(len(result)):
    if result[i] == y_test[i]:
        total_correct += 1
acc = total_correct / len(y_test)
print("acc: {0:.4f}".format(acc))
