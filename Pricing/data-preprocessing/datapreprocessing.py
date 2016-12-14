import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt

train_data = pd.read_csv('../../train.csv', index_col='Id')
# train_data['MSSubClass'].value_counts().plot(kind='barh',rot = 0)
# plt.show()

# arr = np.array([0,1,2,3,4,5])
# index & slic
# print(train_data.index)
# print(train_data.columns)
#demos for some data preprocessing
#slice
slice = train_data.ix[1:4][['MSSubClass', 'MSZoning', 'LotArea']]
slice = train_data.ix[1:4, ['MSSubClass', 'MSZoning', 'LotArea']]# the same
slice = train_data[1:4][['MSSubClass', 'MSZoning', 'LotArea']]#the same,train_data.loc can also be, just refer to the guide
print(slice)

#delete columns
train_data = train_data.drop('MSSubClass', axis=1)# delete one column
train_data = train_data.drop(['MSZoning','MSZoning'],axis=1)# delete two or more columns
#delete rows
train_data = train_data.drop(4)

#value for null data
train_data.loc[train_data[:]['BsmtQual'].isnull(),'BsmtQual'] = 'NoBsmt'

#vector

def preprocess_train():
    train_data = pd.read_csv('../../train.csv', index_col='Id')
    return train_data


