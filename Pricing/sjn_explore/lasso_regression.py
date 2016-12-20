__author__ = 'cat'
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import math
# from dataImport import read_train, na_cols
from collections import Counter
from scipy.stats import skew

import matplotlib.pyplot as plt
import matplotlib
#%%
train_file = u'../../train.csv'
test_file = u'../../test.csv'

train_df = pd.read_csv(train_file, index_col=False)
test_df = pd.read_csv(test_file, index_col=False)

#%%

train_df.drop(['Street', 'Utilities', 'Condition2', 'RoofMatl', 'Alley',
               'GarageYrBlt', 'GarageCond', 'PoolQC', 'MiscFeature','Id'],
              axis=1, inplace=True)

test_df.drop(['Street', 'Utilities', 'Condition2', 'RoofMatl', 'Alley',
               'GarageYrBlt', 'GarageCond', 'PoolQC', 'MiscFeature','Id'],
              axis=1, inplace=True)
all_df = train_df.append(test_df,ignore_index = True)

#%%

def most_commen_value(df, col, compare_dict):
    bool_array = np.array([df[c_name]==c_value for c_name,c_value in compare_dict.items()])
    counter = Counter(df.loc[bool_array.all(axis=0),col])
    return counter.most_common(1)[0]
    
def fill_na_with_common(df, ind, col, compare_cols):
    compare_dict = dict(df.loc[ind,compare_cols])
    
    common_value, common_count = most_commen_value(df, col, compare_dict)
    df.loc[ind, col] = common_value

def get_partial_na_rows(df, col, cols):
    bool_cols = np.array([df[col_name].isnull() for col_name in cols])
    xor_rows = bool_cols.any(axis=0) != bool_cols.all(axis=0)
    return df.loc[np.logical_and(xor_rows, df[col].isnull()),:].index.values

def fill_bsmt_missing(df):
    # marked as NA in BsmtExposure and not NA in other Bsmt Attributes
    bsmt_cols = np.array(['BsmtCond','BsmtExposure','BsmtFinType1','BsmtQual','BsmtFinType2'])

    # fill partial NA with most common values
    for col in bsmt_cols:
        partial_na_rows = get_partial_na_rows(df,col, bsmt_cols)
        for ind in partial_na_rows:
            fill_na_with_common(df, ind, col, bsmt_cols[bsmt_cols!=col])
            
    # fill all NA with 'NoBsmt'
    df.fillna(dict([[col_name,'NoBsmt'] for col_name in bsmt_cols]),inplace=True)
    return df

#%%
def pre_process(df):
    # LotFrontage's N/A is assigned zero, will it cause problem?
    df.fillna(value={'MasVnrType': 'None', 'MasVnrArea': 0,'Electrical': 'SBrkr', 'FireplaceQu': 'NoFP', 'GarageType': 'Noga',
                           'GarageFinish': 'Noga', 'GarageQual': 'Noga', 'Fence': 'NoFence', 
                           'BsmtFinSF1':0,'BsmtFinSF2':0,'BsmtUnfSF':0,'TotalBsmtSF':0,'BsmtFullBath':0,'BsmtHalfBath':0,
                           'LotFrontage': 0},
                    inplace=True)
    
    df.loc[:, 'YrSold'] = 2016 - df.loc[:, 'YrSold']
    
    df.loc[df.loc[:, 'PoolArea'] != 0, 'PoolArea'] = 1
    
    df.loc[:, 'Porch'] = np.sum(df.loc[:, ['EnclosedPorch', '3SsnPorch', 'ScreenPorch']], axis=1)
    df.drop(['EnclosedPorch', '3SsnPorch', 'ScreenPorch'], axis=1, inplace=True)
    
    df.replace({'BsmtFullBath': {3: 2}, 'LotShape': {'IR3': 'IR2'}}, inplace=True)
    
    
    # fill missing values in bsmt
    df = fill_bsmt_missing(df)
    
    def fill_na(df, col_name, value = None):
        if value == None:
            value = df[col_name].mean()
        df.loc[df[col_name].isnull(),col_name] = value
    
    fill_na(df, 'Fence','WD')
    fill_na(df, 'GarageArea')
    fill_na(df, 'GarageCars')
    fill_na(df, 'SaleType', df['SaleType'].mode().values[0])
    fill_na(df, 'KitchenQual', df['KitchenQual'].mode().values[0])
    fill_na(df, 'Functional', df['Functional'].mode().values[0])
    fill_na(df, 'Exterior1st', df['Exterior1st'].mode().values[0])
    fill_na(df, 'Exterior2nd', df['Exterior2nd'].mode().values[0])
    fill_na(df, 'MSZoning', 'RL')

    
    bool_cols = np.array([df[col_name].isnull() for col_name in df.columns])
    print('rows containing na:',np.sum(bool_cols.any(axis=0)))
    print('rows all na:',np.sum(bool_cols.all(axis=0)))
    
    
    # log1pskewed_feats
    numeric_feats = df.dtypes[df.dtypes != "object"].index

    skewed_feats = df[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
    skewed_feats = skewed_feats[skewed_feats > 0.75]
    skewed_feats = skewed_feats.index
    df[skewed_feats] = np.log1p(df[skewed_feats])

    
    return df

#%%
#log transform the target: ignore for test data
#

#train_data = pre_process(train_df.copy())
#test_data = pre_process(test_df.copy())

all_data = pre_process(all_df.ix[:,all_df.columns!='SalePrice'].copy())
all_data = pd.get_dummies(all_data)

#%%
#train_data = pd.get_dummies(train_data)
#test_data = pd.get_dummies(test_data)

X_train = all_data.ix[:train_df.shape[0]-1,all_data.columns!='SalePrice']
y_train = np.log1p(all_df.ix[:train_df.shape[0]-1,'SalePrice'].copy())
X_test = all_data.ix[train_df.shape[0]:,all_data.columns!='SalePrice']

#%%
def rmsle(y, y_pred):
	assert len(y) == len(y_pred)
	terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
	return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5

#%%
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score

#%%
def rmse_cv(model, X , y):
    rmse= np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)
    
    
#%%
def coef_analysis(model_name,model, X):
    coef = pd.Series(model.coef_, index = X.columns)

    print(model_name+"picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

    imp_coef = pd.concat([coef.sort_values().head(10),
                          coef.sort_values().tail(10)])


    
    
#%%
def ridge_train(X,y):
    alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
    cv_ridge = [rmse_cv(Ridge(alpha = alpha),X,y).mean() for alpha in alphas]
    cv_ridge = pd.Series(cv_ridge, index = alphas)
    cv_ridge.plot(title = "Validation - Just Do It")

    print ('min cv is : ',cv_ridge.min())
    
    return alphas[cv_ridge.values.argmin()]

#%%
# ridge regression doesn't remove any property
ridge_alpha = ridge_train(X_train,y_train)
ridge_model = Ridge(alpha = ridge_alpha)

#%%
def lasso_train(X, y):
    model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X, y)
    print ('lasso mean cv is ',rmse_cv(model_lasso,X,y).mean())
    
    return model_lasso
    
#%%
lasso_model = lasso_train(X_train,y_train)

coef_analysis('Lasso', lasso_model, X_train)

rmsle(np.exp(lasso_model.predict(X_train)),np.exp(y_train.values))

#%%
# try to use only useful attributes
lasso_coef = pd.Series(lasso_model.coef_, index = X_train.columns)
lasso_useful_cols = lasso_coef[lasso_coef != 0].index.values
X_train_lasso_useful = X_train.loc[:,lasso_useful_cols]
lasso_model_pruned = lasso_train(X_train_lasso_useful,y_train)
coef_analysis('Lasso Pruned',lasso_model_pruned, X_train_lasso_useful)
rmsle(np.exp(lasso_model_pruned.predict(X_train_lasso_useful)),np.exp(y_train.values))


#%%
preds = lasso_model_pruned.predict(X_test.loc[:,lasso_useful_cols])
#%%
prices = pd.Series(np.exp(preds)-1,index=(X_test.index.values+1))
#%%
print('GrLivArea:\n',X_test['GrLivArea'].values,'pred:\n',np.exp(preds)-1)
plt.scatter(list(X_test['GrLivArea'].values), list((np.exp(preds)-1)))
plt.title("Looking for outliers")
plt.xlabel("GrLivArea")
plt.ylabel("SalePrice")
plt.show()
prices.to_csv('../../pred_prices.csv')