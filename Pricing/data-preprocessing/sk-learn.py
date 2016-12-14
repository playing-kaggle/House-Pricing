import pandas as pd
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt

train_data = pd.read_csv('../../train.csv', index_col='Id')

# print(unique_value)

'''
    vectorize using some column MSZoning as an example
'''


column_list = ['MSSubClass', 'MSZoning', 'LotShape', 'LandContour']
new_column_list = []
for column in column_list:
    unique_value = train_data[column].unique()
    for value in unique_value:
        new_column_name = str(column) + '_' + str(value)
        train_data.loc[train_data[column] == value, new_column_name] = 1
        train_data.loc[train_data[column] != value, new_column_name] = 0
        new_column_list.append(new_column_name)
    train_data.drop(column,inplace=True)
#print(train_data.columns)
#print(train_data[new_column_list])
new_column_list.extend(['BsmtFinSF1','SalePrice'])

train_data = train_data[new_column_list]
data_X = train_data.drop('SalePrice', axis=1)
data_Y = train_data['SalePrice']
#print(data_X['BsmtFinSF1'])
#data_X = np.asarray(data_X)
#data_Y = np.asarray(data_Y)
#print(data_X);print(data_Y)
regr = linear_model.LinearRegression()
regr.fit(data_X, data_Y)
print('Coefficients: \n', regr.coef_)
print(regr.predict(data_X))
print("Mean squared error: %.2f"
      % np.mean((regr.predict(data_X) - data_Y) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(data_X, data_Y))


#print(data)
#plt.plot(data_Y, regr.predict(data_X), color='blue',
#        linewidth=3)
plt.scatter(data_Y,regr.predict(data_X))
plt.xticks(())
plt.yticks(())
plt.show()
# print(train_data.columns)
