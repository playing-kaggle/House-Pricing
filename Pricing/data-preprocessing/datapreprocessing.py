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


def vectorize(data):
    nomial_list = ['MSSubClass', 'MSZoning', 'LotShape', 'LandContour', 'LotConfig', 'LandSlope',
                   'Neighborhood', 'Condition1', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'RoofStyle',
                   'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
                   'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating',
                   'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu',
                   'GarageType', 'GarageFinish', 'GarageQual', 'PavedDrive', 'PoolArea', 'Fence', 'SaleType',
                   'SaleCondition']
    for column in nomial_list:
        unique_value = data[column].unique()
        for value in unique_value:
            new_column_name = str(column) + '_' + str(value)
            data.loc[data[column] == value, new_column_name] = 1
            data.loc[data[column] != value, new_column_name] = 0
        del data[column]


def standardize(data):

    numeric_list = ['BsmtFullBath', 'LotArea', 'YearRemodAdd', 'GrLivArea', 'BsmtHalfBath', 'MiscVal', 'YearBuilt',
                    'WoodDeckSF', 'KitchenAbvGr', 'TotalBsmtSF', 'GarageArea', 'GarageCars', 'OpenPorchSF', 'MoSold',
                    'LowQualFinSF', 'BedroomAbvGr', 'Fireplaces', '1stFlrSF', 'FullBath', 'BsmtFinSF1', 'BsmtFinSF2',
                    'HalfBath',
                    'Porch', '2ndFlrSF', 'MasVnrArea', 'YrSold', 'BsmtUnfSF', 'LotFrontage', 'TotRmsAbvGrd']

    data.loc[:, numeric_list] = preprocessing.scale(data.loc[:, numeric_list])


def build_model():
    # when building a model, it is better to join train_data and test data together
    train_data = data_cleaning('../../train.csv')
    test_data = data_cleaning('../../test.csv')

    test_data.fillna(value={'BsmtFinSF1': 0, 'BsmtFinSF2': 0, 'BsmtUnfSF': 0,
                            'TotalBsmtSF': 0, 'BsmtFullBath': 0, 'BsmtHalfBath': 0,
                            'GarageCars': 0, 'GarageArea': 0}, inplace=True)
    salePrice = train_data['SalePrice']
    total_data = train_data.drop('SalePrice', axis=1).append(test_data)
    total_data.drop('Id',inplace=True)
    print(np.shape(total_data))
    vectorize(total_data)
    standardize(total_data)
    train_data = total_data[0:1460]
    test_data = total_data[1460:]
    regr = linear_model.LinearRegression()
    regr.fit(train_data,salePrice)
    print('Coefficients: \n', regr.coef_)
    print("Mean squared error: %.2f"
              % np.mean((regr.predict(train_data) - salePrice) ** 2))
        # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % regr.score(train_data, salePrice))

    # plt.scatter(salePrice, regr.predict(train_data))
    #
    # plt.show()
    return regr, test_data





def predict_test_data():
    regr,test_data = build_model()
    print(test_data.columns[np.sum(test_data.isnull(), axis=0) != 0])
    result = regr.predict(test_data)
    print('result is null :',sum(result == np.nan))
    print('length: \n', len(result))

    df = pd.DataFrame(np.abs(np.round(result,1)),index=list(range(1461,2920)),columns=['SalePrice'])
    df.to_csv('../../submission1214.csv')
    print('result: \n', df)
    #predict = regr.predict(test_data)


predict_test_data()
