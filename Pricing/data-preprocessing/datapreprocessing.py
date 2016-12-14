import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model, preprocessing

# train_data = pd.read_csv('../../train.csv', index_col='Id')
# # train_data['MSSubClass'].value_counts().plot(kind='barh',rot = 0)
# # plt.show()
#
# # arr = np.array([0,1,2,3,4,5])
# # index & slic
# # print(train_data.index)
# # print(train_data.columns)
# #demos for some data preprocessing
# #slice
# slice = train_data.ix[1:4][['MSSubClass', 'MSZoning', 'LotArea']]
# slice = train_data.ix[1:4, ['MSSubClass', 'MSZoning', 'LotArea']]# the same
# slice = train_data[1:4][['MSSubClass', 'MSZoning', 'LotArea']]#the same,train_data.loc can also be, just refer to the guide
# print(slice)
#
# #delete columns
# train_data = train_data.drop('MSSubClass', axis=1)# delete one column
# train_data = train_data.drop(['MSZoning','MSZoning'],axis=1)# delete two or more columns
# #delete rows
# train_data = train_data.drop(4)
#
# #value for null data
# train_data.loc[train_data[:]['BsmtQual'].isnull(),'BsmtQual'] = 'NoBsmt'

'''
    data cleaning, handle missing data
'''


def data_cleaning(file_path):
    data = pd.read_csv(file_path, index_col=False)
    data.drop(['Street', 'Utilities', 'Condition2', 'RoofMatl', 'Alley',
               'GarageYrBlt', 'GarageCond', 'PoolQC', 'MiscFeature'],
              axis=1, inplace=True)
    # marked as NA in BsmtExposure and not NA in other Bsmt Attributes
    data.loc[np.logical_xor(data['BsmtCond'].isnull(), data['BsmtExposure'].isnull()), 'BsmtExposure'] = 'No'
    # LotFrontage's N/A is assigned zero, will it cause problem?
    data.fillna(value={'MasVnrType': 'None', 'MasVnrArea': 0, 'BsmtQual': 'NoBsmt', 'BsmtCond': 'NoBsmt',
                       'BsmtExposure': 'NoBsmt', 'BsmtFinType1': 'NoBsmt', 'BsmtFinType2': 'NoBsmt',
                       'Electrical': 'SBrkr', 'FireplaceQu': 'NoFP', 'GarageType': 'Noga',
                       'GarageFinish': 'Noga', 'GarageQual': 'Noga', 'Fence': 'NoFence', 'LotFrontage': 0},
                inplace=True)
    data.loc[:, 'YrSold'] = 2016 - data.loc[:, 'YrSold']
    data.loc[:, 'YearBuilt'] = 2016 - data.loc[:, 'YearBuilt']
    data.loc[:, 'YearRemodAdd'] = 2016 - data.loc[:, 'YearRemodAdd']
    data.loc[data.loc[:, 'PoolArea'] != 0, 'PoolArea'] = 'Y'
    data.loc[data.loc[:, 'PoolArea'] == 0, 'PoolArea'] = 'N'
    data.loc[:, 'Porch'] = np.sum(data.loc[:, ['EnclosedPorch', '3SsnPorch', 'ScreenPorch']], axis=1)
    data.drop(['EnclosedPorch', '3SsnPorch', 'ScreenPorch'], axis=1, inplace=True)
    data.replace({'BsmtFullBath': {3: 2},
                  'LotShape': {'IR3': 'IR2'}},
                 inplace=True)
    return data
    # data.columns
    # examine columns containing NA value
    # print(data)
    # print(data.columns[np.sum(data.isnull(), axis=0) != 0])


'''
    vectorization the dataset and Standardization
'''


def preprocess_train():
    data = data_cleaning('../../train.csv')
    nomial_list = ['MSSubClass', 'MSZoning', 'LotShape', 'LandContour', 'LotConfig', 'LandSlope',
                   'Neighborhood', 'Condition1', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'RoofStyle',
                   'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
                   'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating',
                   'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu',
                   'GarageType', 'GarageFinish', 'GarageQual', 'PavedDrive', 'PoolArea', 'Fence', 'SaleType',
                   'SaleCondition']

    numeric_list = ['BsmtFullBath', 'LotArea', 'YearRemodAdd', 'GrLivArea', 'BsmtHalfBath', 'MiscVal', 'YearBuilt',
                    'WoodDeckSF', 'KitchenAbvGr', 'TotalBsmtSF', 'GarageArea', 'GarageCars', 'OpenPorchSF', 'MoSold',
                    'LowQualFinSF', 'BedroomAbvGr', 'Fireplaces', '1stFlrSF', 'FullBath', 'BsmtFinSF1', 'BsmtFinSF2',
                    'HalfBath',
                    'Porch', '2ndFlrSF', 'MasVnrArea', 'YrSold', 'BsmtUnfSF', 'LotFrontage', 'TotRmsAbvGrd']
    '''
        vectorization nomial attributes
    '''
    for column in nomial_list:
        unique_value = data[column].unique()
        for value in unique_value:
            new_column_name = str(column) + '_' + str(value)
            data.loc[data[column] == value, new_column_name] = 1
            data.loc[data[column] != value, new_column_name] = 0
        del data[column]

    data.loc[:, numeric_list] = preprocessing.scale(data.loc[:, numeric_list])
    return data


def build_model():
    processed_data = preprocess_train()
    processed_data.drop('Id', axis=1, inplace=True)

    print(processed_data)

    data_X = processed_data.drop('SalePrice', axis=1)
    data_Y = processed_data['SalePrice']
    # print(data_X['BsmtFinSF1'])
    # data_X = np.asarray(data_X)
    # data_Y = np.asarray(data_Y)
    # print(data_X);print(data_Y)
    regr = linear_model.LinearRegression()
    regr.fit(data_X, data_Y)
    return data_X,data_Y,regr

data_X,data_Y,regr = build_model()
print('Coefficients: \n', regr.coef_)

print(regr.predict(data_X))
print("Mean squared error: %.2f"
          % np.mean((regr.predict(data_X) - data_Y) ** 2))
    # Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(data_X, data_Y))

    # print(data)
    # plt.plot(data_Y, regr.predict(data_X), color='blue',
    #        linewidth=3)
plt.scatter(data_Y, regr.predict(data_X))
    #plt.xticks(())
    #plt.yticks(())
plt.show()

def predict_test_data():
    test_data = pd.read_csv('../../test.csv', index_col='Id')
    np.sum(test_data.isnull(),axis=0)
    pass
