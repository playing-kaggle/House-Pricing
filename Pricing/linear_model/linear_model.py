# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV,BayesianRidge
from sklearn.metrics import mean_squared_error, make_scorer
from scipy.stats import skew
import matplotlib.pyplot as plt

import math
#A function to calculate Root Mean Squared Logarithmic Error (RMSLE)
def rmsle(y, y_pred):
	return  np.sqrt(mean_squared_error(y,y_pred))



pd.set_option('display.float_format', lambda x: '%.3f' % x)
train = pd.read_csv('../../train.csv')
test = pd.read_csv('../../test.csv')

print(train.shape)

# Looking for outliers, as indicated in https://ww2.amstat.org/publications/jse/v19n3/decock.pdf
# plt.scatter(train.GrLivArea, train.SalePrice, c = "blue", marker = "s")
# plt.title("Looking for outliers")
# plt.xlabel("GrLivArea")
# plt.ylabel("SalePrice")
# plt.show()
train = train[train.GrLivArea < 4000]
train.SalePrice = np.log1p(train.SalePrice)
y = train.SalePrice

# combine train_data and test_data together
train = train.append(test, ignore_index=True)
# Drop  column
train.drop(['Id', 'Street', 'Utilities', 'Condition2', 'RoofMatl', 'Alley',
            'GarageYrBlt', 'MiscFeature'], axis=1, inplace=True)

'''
fill missing data
'''

# BedroomAbvGr : NA most likely means 0
train.loc[:, "BedroomAbvGr"] = train.loc[:, "BedroomAbvGr"].fillna(0)
# BsmtQual etc : data description says NA for basement features is "no basement"
train.loc[:, "BsmtQual"] = train.loc[:, "BsmtQual"].fillna("No")
train.loc[:, "BsmtCond"] = train.loc[:, "BsmtCond"].fillna("No")
train.loc[:, "BsmtExposure"] = train.loc[:, "BsmtExposure"].fillna("No")
train.loc[:, "BsmtFinType1"] = train.loc[:, "BsmtFinType1"].fillna("No")
train.loc[:, "BsmtFinType2"] = train.loc[:, "BsmtFinType2"].fillna("No")
train.loc[:, "BsmtFullBath"] = train.loc[:, "BsmtFullBath"].fillna(0)
train.loc[:, "BsmtHalfBath"] = train.loc[:, "BsmtHalfBath"].fillna(0)
train.loc[:, "BsmtUnfSF"] = train.loc[:, "BsmtUnfSF"].fillna(0)
# CentralAir : NA most likely means No
train.loc[:, "CentralAir"] = train.loc[:, "CentralAir"].fillna("N")
# Condition : NA most likely means Normal
train.loc[:, "Condition1"] = train.loc[:, "Condition1"].fillna("Norm")
# EnclosedPorch : NA most likely means no enclosed porch
train.loc[:, "EnclosedPorch"] = train.loc[:, "EnclosedPorch"].fillna(0)
# External stuff : NA most likely means average
train.loc[:, "ExterCond"] = train.loc[:, "ExterCond"].fillna("TA")
train.loc[:, "ExterQual"] = train.loc[:, "ExterQual"].fillna("TA")
# Fence : data description says NA means "no fence"
train.loc[:, "Fence"] = train.loc[:, "Fence"].fillna("No")
# FireplaceQu : data description says NA means "no fireplace"
train.loc[:, "FireplaceQu"] = train.loc[:, "FireplaceQu"].fillna("No")
train.loc[:, "Fireplaces"] = train.loc[:, "Fireplaces"].fillna(0)
# Functional : data description says NA means typical
train.loc[:, "Functional"] = train.loc[:, "Functional"].fillna("Typ")
# GarageType etc : data description says NA for garage features is "no garage"
train.loc[:, "GarageType"] = train.loc[:, "GarageType"].fillna("No")
train.loc[:, "GarageFinish"] = train.loc[:, "GarageFinish"].fillna("No")
train.loc[:, "GarageQual"] = train.loc[:, "GarageQual"].fillna("No")
train.loc[:, "GarageCond"] = train.loc[:, "GarageCond"].fillna("No")
train.loc[:, "GarageArea"] = train.loc[:, "GarageArea"].fillna(0)
train.loc[:, "GarageCars"] = train.loc[:, "GarageCars"].fillna(0)
# HalfBath : NA most likely means no half baths above grade
train.loc[:, "HalfBath"] = train.loc[:, "HalfBath"].fillna(0)
# HeatingQC : NA most likely means typical
train.loc[:, "HeatingQC"] = train.loc[:, "HeatingQC"].fillna("TA")
# KitchenAbvGr : NA most likely means 0
train.loc[:, "KitchenAbvGr"] = train.loc[:, "KitchenAbvGr"].fillna(0)
# KitchenQual : NA most likely means typical
train.loc[:, "KitchenQual"] = train.loc[:, "KitchenQual"].fillna("TA")
# LotFrontage : NA most likely means no lot frontage
train.loc[:, "LotFrontage"] = train.loc[:, "LotFrontage"].fillna(0)
# LotShape : NA most likely means regular
train.loc[:, "LotShape"] = train.loc[:, "LotShape"].fillna("Reg")
# MasVnrType : NA most likely means no veneer
train.loc[:, "MasVnrType"] = train.loc[:, "MasVnrType"].fillna("None")
train.loc[:, "MasVnrArea"] = train.loc[:, "MasVnrArea"].fillna(0)
train.loc[:, "MiscVal"] = train.loc[:, "MiscVal"].fillna(0)
# OpenPorchSF : NA most likely means no open porch
train.loc[:, "OpenPorchSF"] = train.loc[:, "OpenPorchSF"].fillna(0)
# PavedDrive : NA most likely means not paved
train.loc[:, "PavedDrive"] = train.loc[:, "PavedDrive"].fillna("N")
# PoolQC : data description says NA means "no pool"
train.loc[:, "PoolQC"] = train.loc[:, "PoolQC"].fillna("No")
train.loc[:, "PoolArea"] = train.loc[:, "PoolArea"].fillna(0)
# SaleCondition : NA most likely means normal sale
train.loc[:, "SaleCondition"] = train.loc[:, "SaleCondition"].fillna("Normal")
# ScreenPorch : NA most likely means no screen porch
train.loc[:, "ScreenPorch"] = train.loc[:, "ScreenPorch"].fillna(0)
# TotRmsAbvGrd : NA most likely means 0
train.loc[:, "TotRmsAbvGrd"] = train.loc[:, "TotRmsAbvGrd"].fillna(0)

# WoodDeckSF : NA most likely means no wood deck
train.loc[:, "WoodDeckSF"] = train.loc[:, "WoodDeckSF"].fillna(0)

# Some numerical features are actually really categories
train = train.replace({"MSSubClass": {20: "SC20", 30: "SC30", 40: "SC40", 45: "SC45",
                                      50: "SC50", 60: "SC60", 70: "SC70", 75: "SC75",
                                      80: "SC80", 85: "SC85", 90: "SC90", 120: "SC120",
                                      150: "SC150", 160: "SC160", 180: "SC180", 190: "SC190"},
                       "MoSold": {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                                  7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
                       })

# Encode some ordered categorical features as ordered numbers
train = train.replace({
    "BsmtCond": {"No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "BsmtExposure": {"No": 0, "Mn": 1, "Av": 2, "Gd": 3},
    "BsmtFinType1": {"No": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4,
                     "ALQ": 5, "GLQ": 6},
    "BsmtFinType2": {"No": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4,
                     "ALQ": 5, "GLQ": 6},
    "BsmtQual": {"No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "ExterCond": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "ExterQual": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "FireplaceQu": {"No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "Functional": {"Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4, "Mod": 5,
                   "Min2": 6, "Min1": 7, "Typ": 8},
    "GarageCond": {"No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "GarageQual": {"No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "HeatingQC": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "KitchenQual": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "LandSlope": {"Sev": 1, "Mod": 2, "Gtl": 3},
    "LotShape": {"IR3": 1, "IR2": 2, "IR1": 3, "Reg": 4},
    "PavedDrive": {"N": 0, "P": 1, "Y": 2},
    "PoolQC": {"No": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},

}
)
# handle year
train.loc[:, 'YrSold'] = 2016 - train.loc[:, 'YrSold']
train.loc[:, 'YearBuilt'] = 2016 - train.loc[:, 'YearBuilt']
train.loc[:, 'YearRemodAdd'] = 2016 - train.loc[:, 'YearRemodAdd']

'''
 Create new features
'''
# 1* Simplifications of existing features
train["SimplOverallQual"] = train.OverallQual.replace({1: 1, 2: 1, 3: 1,  # bad
                                                       4: 2, 5: 2, 6: 2,  # average
                                                       7: 3, 8: 3, 9: 3, 10: 3  # good
                                                       })
train["SimplOverallCond"] = train.OverallCond.replace({1: 1, 2: 1, 3: 1,  # bad
                                                       4: 2, 5: 2, 6: 2,  # average
                                                       7: 3, 8: 3, 9: 3, 10: 3  # good
                                                       })
train["SimplPoolQC"] = train.PoolQC.replace({1: 1, 2: 1,  # average
                                             3: 2, 4: 2  # good
                                             })
train["SimplGarageCond"] = train.GarageCond.replace({1: 1,  # bad
                                                     2: 1, 3: 1,  # average
                                                     4: 2, 5: 2  # good
                                                     })
train["SimplGarageQual"] = train.GarageQual.replace({1: 1,  # bad
                                                     2: 1, 3: 1,  # average
                                                     4: 2, 5: 2  # good
                                                     })
train["SimplFireplaceQu"] = train.FireplaceQu.replace({1: 1,  # bad
                                                       2: 1, 3: 1,  # average
                                                       4: 2, 5: 2  # good
                                                       })
train["SimplFireplaceQu"] = train.FireplaceQu.replace({1: 1,  # bad
                                                       2: 1, 3: 1,  # average
                                                       4: 2, 5: 2  # good
                                                       })
train["SimplFunctional"] = train.Functional.replace({1: 1, 2: 1,  # bad
                                                     3: 2, 4: 2,  # major
                                                     5: 3, 6: 3, 7: 3,  # minor
                                                     8: 4  # typical
                                                     })
train["SimplKitchenQual"] = train.KitchenQual.replace({1: 1,  # bad
                                                       2: 1, 3: 1,  # average
                                                       4: 2, 5: 2  # good
                                                       })
train["SimplHeatingQC"] = train.HeatingQC.replace({1: 1,  # bad
                                                   2: 1, 3: 1,  # average
                                                   4: 2, 5: 2  # good
                                                   })
train["SimplBsmtFinType1"] = train.BsmtFinType1.replace({1: 1,  # unfinished
                                                         2: 1, 3: 1,  # rec room
                                                         4: 2, 5: 2, 6: 2  # living quarters
                                                         })
train["SimplBsmtFinType2"] = train.BsmtFinType2.replace({1: 1,  # unfinished
                                                         2: 1, 3: 1,  # rec room
                                                         4: 2, 5: 2, 6: 2  # living quarters
                                                         })
train["SimplBsmtCond"] = train.BsmtCond.replace({1: 1,  # bad
                                                 2: 1, 3: 1,  # average
                                                 4: 2, 5: 2  # good
                                                 })
train["SimplBsmtQual"] = train.BsmtQual.replace({1: 1,  # bad
                                                 2: 1, 3: 1,  # average
                                                 4: 2, 5: 2  # good
                                                 })
train["SimplExterCond"] = train.ExterCond.replace({1: 1,  # bad
                                                   2: 1, 3: 1,  # average
                                                   4: 2, 5: 2  # good
                                                   })
train["SimplExterQual"] = train.ExterQual.replace({1: 1,  # bad
                                                   2: 1, 3: 1,  # average
                                                   4: 2, 5: 2  # good
                                                   })

# 2* Combinations of existing features
# Overall quality of the house
train["OverallGrade"] = train["OverallQual"] * train["OverallCond"]
# Overall quality of the garage
train["GarageGrade"] = train["GarageQual"] * train["GarageCond"]
# Overall quality of the exterior
train["ExterGrade"] = train["ExterQual"] * train["ExterCond"]
# Overall kitchen score
train["KitchenScore"] = train["KitchenAbvGr"] * train["KitchenQual"]
# Overall fireplace score
train["FireplaceScore"] = train["Fireplaces"] * train["FireplaceQu"]
# Overall garage score
train["GarageScore"] = train["GarageArea"] * train["GarageQual"]
# Overall pool score
train["PoolScore"] = train["PoolArea"] * train["PoolQC"]
# Simplified overall quality of the house
train["SimplOverallGrade"] = train["SimplOverallQual"] * train["SimplOverallCond"]
# Simplified overall quality of the exterior
train["SimplExterGrade"] = train["SimplExterQual"] * train["SimplExterCond"]
# Simplified overall pool score
train["SimplPoolScore"] = train["PoolArea"] * train["SimplPoolQC"]
# Simplified overall garage score
train["SimplGarageScore"] = train["GarageArea"] * train["SimplGarageQual"]
# Simplified overall fireplace score
train["SimplFireplaceScore"] = train["Fireplaces"] * train["SimplFireplaceQu"]
# Simplified overall kitchen score
train["SimplKitchenScore"] = train["KitchenAbvGr"] * train["SimplKitchenQual"]
# Total number of bathrooms
train["TotalBath"] = train["BsmtFullBath"] + (0.5 * train["BsmtHalfBath"]) + \
                     train["FullBath"] + (0.5 * train["HalfBath"])
# Total SF for house (incl. basement)
train["AllSF"] = train["GrLivArea"] + train["TotalBsmtSF"]
# Total SF for 1st + 2nd floors
train["AllFlrsSF"] = train["1stFlrSF"] + train["2ndFlrSF"]
# Total SF for porch
train["AllPorchSF"] = train["OpenPorchSF"] + train["EnclosedPorch"] + \
                      train["3SsnPorch"] + train["ScreenPorch"]
# Has masonry veneer or not
train["HasMasVnr"] = train.MasVnrType.replace({"BrkCmn": 1, "BrkFace": 1, "CBlock": 1,
                                               "Stone": 1, "None": 0})
# House completed before sale or not
train["BoughtOffPlan"] = train.SaleCondition.replace({"Abnorml": 0, "Alloca": 0, "AdjLand": 0,
                                                      "Family": 0, "Normal": 0, "Partial": 1})



# Create new features
# 3* Polynomials on the top 10 existing features
train["OverallQual-s2"] = train["OverallQual"] ** 2
train["OverallQual-s3"] = train["OverallQual"] ** 3
train["OverallQual-Sq"] = np.sqrt(train["OverallQual"])
train["AllSF-2"] = train["AllSF"] ** 2
train["AllSF-3"] = train["AllSF"] ** 3
train["AllSF-Sq"] = np.sqrt(train["AllSF"])
train["AllFlrsSF-2"] = train["AllFlrsSF"] ** 2
train["AllFlrsSF-3"] = train["AllFlrsSF"] ** 3
train["AllFlrsSF-Sq"] = np.sqrt(train["AllFlrsSF"])
train["GrLivArea-2"] = train["GrLivArea"] ** 2
train["GrLivArea-3"] = train["GrLivArea"] ** 3
train["GrLivArea-Sq"] = np.sqrt(train["GrLivArea"])
train["SimplOverallQual-s2"] = train["SimplOverallQual"] ** 2
train["SimplOverallQual-s3"] = train["SimplOverallQual"] ** 3
train["SimplOverallQual-Sq"] = np.sqrt(train["SimplOverallQual"])
train["ExterQual-2"] = train["ExterQual"] ** 2
train["ExterQual-3"] = train["ExterQual"] ** 3
train["ExterQual-Sq"] = np.sqrt(train["ExterQual"])
train["GarageCars-2"] = train["GarageCars"] ** 2
train["GarageCars-3"] = train["GarageCars"] ** 3
train["GarageCars-Sq"] = np.sqrt(train["GarageCars"])
train["TotalBath-2"] = train["TotalBath"] ** 2
train["TotalBath-3"] = train["TotalBath"] ** 3
train["TotalBath-Sq"] = np.sqrt(train["TotalBath"])
train["KitchenQual-2"] = train["KitchenQual"] ** 2
train["KitchenQual-3"] = train["KitchenQual"] ** 3
train["KitchenQual-Sq"] = np.sqrt(train["KitchenQual"])
train["GarageScore-2"] = train["GarageScore"] ** 2
train["GarageScore-3"] = train["GarageScore"] ** 3
train["GarageScore-Sq"] = np.sqrt(train["GarageScore"])

# Differentiate numerical features (minus the target) and categorical features
categorical_features = train.select_dtypes(include=["object"]).columns
numerical_features = train.select_dtypes(exclude=["object"]).columns
numerical_features = numerical_features.drop("SalePrice")
print("Numerical features : " + str(len(numerical_features)))
print("Categorical features : " + str(len(categorical_features)))
train_num = train[numerical_features]
train_cat = train[categorical_features]

print("NAs for numerical features in train : " + str(train_num.isnull().values.sum()))
train_num = train_num.fillna(train_num.median())
print("Remaining NAs for numerical features in train : " + str(train_num.isnull().values.sum()))

# Log transform of the skewed numerical features to lessen impact of outliers
# Inspired by Alexandru Papiu's script : https://www.kaggle.com/apapiu/house-prices-advanced-regression-techniques/regularized-linear-models
# As a general rule of thumb, a skewness with an absolute value > 0.5 is considered at least moderately skewed
skewness = train_num.apply(lambda x: skew(x))
skewness = skewness[abs(skewness) > 0.5]
print(str(skewness.shape[0]) + " skewed numerical features to log transform")
skewed_features = skewness.index
train_num[skewed_features] = np.log1p(train_num[skewed_features])

print("NAs for categorical features in train : " + str(train_cat.isnull().values.sum()))
train_cat = pd.get_dummies(train_cat)
print("Remaining NAs for categorical features in train : " + str(train_cat.isnull().values.sum()))

# Join categorical and numerical features
train = pd.concat([train_num, train_cat], axis=1)

# Partition the dataset in train + validation sets

# print("y_test : " + str(y_test.shape))
# standardize numeric attributes
stdSc = StandardScaler()
train.loc[:, numerical_features] = stdSc.fit_transform(train.loc[:, numerical_features])
# split from combined data
train_split = train[:(train.shape[0] - test.shape[0])]
print("Find most important features relative to target")
corr = pd.concat([train_split,y],axis=1).corr()
drop_columns = list(corr['SalePrice'].loc[corr['SalePrice'] < 0,].index)
print('old number of features:',str(train.shape[1]))
print('drop columns:',drop_columns)
train.drop(drop_columns,axis=1, inplace=True)
print("New number of features : " + str(train.shape[1]))
train_split = train[:(train.shape[0] - test.shape[0])]
test = train[(0 - test.shape[0]):]
X_train, X_test, y_train, y_test = train_test_split(train_split, y, test_size=0.3, random_state=0)


scorer = make_scorer(mean_squared_error, greater_is_better=False)
def rmse_cv(model, X, Y):
    rmse = np.sqrt(-cross_val_score(model, X, Y, scoring=scorer, cv=10))
    return (rmse)


def linear_regression():
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    # Look at predictions on training and validation set
    print("RMSE on Training set :", rmse_cv(lr, train_split, y).mean())
    y_train_pred = lr.predict(train_split)
    print('rmsle calculate by self:', rmsle(list(np.exp(y) - 1), list(np.exp(y_train_pred) - 1)))
    plt.scatter(y_train_pred, y_train_pred - y, c="blue", marker="s", label="Training data")
    plt.title("Linear regression")
    plt.xlabel("Predicted values")
    plt.ylabel("Residuals")
    plt.legend(loc="upper left")
    plt.hlines(y=0, xmin=10.5, xmax=13.5, color="red")
    plt.show()
    # Plot predictions
    plt.scatter(y_train_pred, y, c="blue", marker="s", label="Training data")
    plt.title("Linear regression")
    plt.xlabel("Predicted values")
    plt.ylabel("Real values")
    plt.legend(loc="upper left")
    plt.plot([10.5, 13.5], [10.5, 13.5], c="red")
    plt.show()
    return lr


def ridge_regression():
    ridge = RidgeCV(alphas=[0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60])
    ridge.fit(X_train, y_train)
    alpha = ridge.alpha_
    print("Best alpha :", alpha)
    print("Try again for more precision with alphas centered around " + str(alpha))
    ridge = RidgeCV(alphas=[alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85,
                            alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,
                            alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4],
                    cv=10)
    ridge.fit(X_train, y_train)
    alpha = ridge.alpha_
    print("Best alpha :", alpha)
    print("Ridge RMSE on Training set :", rmse_cv(ridge, X_train, y_train).mean())
    print("Ridge RMSE on Test set :", rmse_cv(ridge, X_test, y_test).mean())
    y_train_rdg = ridge.predict(X_train)
    y_test_rdg = ridge.predict(X_test)
    # Plot residuals
    plt.scatter(y_train_rdg, y_train_rdg - y_train, c="blue", marker="s", label="Training data")
    plt.scatter(y_test_rdg, y_test_rdg - y_test, c="lightgreen", marker="s", label="Validation data")
    plt.title("Linear regression with Ridge regularization")
    plt.xlabel("Predicted values")
    plt.ylabel("Residuals")
    plt.legend(loc="upper left")
    plt.hlines(y=0, xmin=10.5, xmax=13.5, color="red")
    plt.show()
    # Plot predictions
    plt.scatter(y_train_rdg, y_train, c="blue", marker="s", label="Training data")
    plt.scatter(y_test_rdg, y_test, c="lightgreen", marker="s", label="Validation data")
    plt.title("Linear regression with Ridge regularization")
    plt.xlabel("Predicted values")
    plt.ylabel("Real values")
    plt.legend(loc="upper left")
    plt.plot([10.5, 13.5], [10.5, 13.5], c="red")
    plt.show()
    # Plot important coefficients
    coefs = pd.Series(ridge.coef_, index=X_train.columns)
    print("Ridge picked " + str(sum(coefs != 0)) + " features and eliminated the other " + \
          str(sum(coefs == 0)) + " features")
    imp_coefs = pd.concat([coefs.sort_values().head(10),
                           coefs.sort_values().tail(10)])
    imp_coefs.plot(kind="barh")
    plt.title("Coefficients in the Ridge Model")
    plt.show()

    return ridge


def Lasso_regression():
    lasso = LassoCV(alphas=[0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1,
                            0.3, 0.6, 1],
                    max_iter=50000, cv=10)
    lasso.fit(train_split, y)
    alpha = lasso.alpha_
    print("Best alpha :", alpha)
    print("Try again for more precision with alphas centered around " + str(alpha))
    lasso = LassoCV(alphas=[alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8,
                            alpha * .85, alpha * .9, alpha * .95, alpha, alpha * 1.05,
                            alpha * 1.1, alpha * 1.15, alpha * 1.25, alpha * 1.3, alpha * 1.35,
                            alpha * 1.4],
                    max_iter=50000, cv=10)
    lasso.fit(train_split, y)
    alpha = lasso.alpha_
    print("Best alpha :", alpha)
    print("Lasso RMSE on Training set :", rmse_cv(lasso, train_split, y).mean())

    y_train_las = lasso.predict(train_split)
    # Plot residuals
    plt.scatter(y_train_las, y_train_las - y, c="blue", marker="s", label="Training data")
    plt.title("Linear regression with Lasso regularization")
    plt.xlabel("Predicted values")
    plt.ylabel("Residuals")
    plt.legend(loc="upper left")
    plt.hlines(y=0, xmin=10.5, xmax=13.5, color="red")
    plt.show()
    # Plot predictions
    plt.scatter(y_train_las, y, c="blue", marker="s", label="Training data")
    plt.title("Linear regression with Lasso regularization")
    plt.xlabel("Predicted values")
    plt.ylabel("Real values")
    plt.legend(loc="upper left")
    plt.plot([10.5, 13.5], [10.5, 13.5], c="red")
    plt.show()
    # # Plot important coefficients
    coefs = pd.DataFrame(lasso.coef_, index=X_train.columns,columns=['value'])
    # print("Lasso picked " + str(sum(coefs != 0)) + " features and eliminated the other " + \
    #       str(sum(coefs == 0)) + " features")
    # imp_coefs = pd.concat([coefs.sort_values().head(10),
    #                        coefs.sort_values().tail(10)])
    # imp_coefs.plot(kind="barh")
    # plt.title("Coefficients in the Lasso Model")
    # plt.show()

    return coefs,lasso


def Elasticnet_regression(X=train_split,Y=y):
    elasticNet = ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1],
                              alphas=[0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006,
                                      0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6],
                              max_iter=50000, cv=10)
    print('handled data columns :\n', train_split.columns)
    elasticNet.fit(X, Y)
    alpha = elasticNet.alpha_
    ratio = elasticNet.l1_ratio_
    print("Best l1_ratio :", ratio)
    print("Best alpha :", alpha)
    print("Try again for more precision with l1_ratio centered around " + str(ratio))
    elasticNet = ElasticNetCV(
        l1_ratio=[ratio * .85, ratio * .9, ratio * .95, ratio, ratio * 1.05, ratio * 1.1, ratio * 1.15],
        alphas=[0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6],
        max_iter=50000, cv=10)
    elasticNet.fit(X, Y)
    if (elasticNet.l1_ratio_ > 1):
        elasticNet.l1_ratio_ = 1
    alpha = elasticNet.alpha_
    ratio = elasticNet.l1_ratio_
    print("Best l1_ratio :", ratio)
    print("Best alpha :", alpha)
    print("Now try again for more precision on alpha, with l1_ratio fixed at " + str(ratio) +
          " and alpha centered around " + str(alpha))
    elasticNet = ElasticNetCV(l1_ratio=ratio,
                              alphas=[alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85,
                                      alpha * .9,
                                      alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15, alpha * 1.25,
                                      alpha * 1.3,
                                      alpha * 1.35, alpha * 1.4],
                              max_iter=50000, cv=10)
    elasticNet.fit(X, Y)
    if (elasticNet.l1_ratio_ > 1):
        elasticNet.l1_ratio_ = 1
    alpha = elasticNet.alpha_
    ratio = elasticNet.l1_ratio_
    print("Best l1_ratio :", ratio)
    print("Best alpha :", alpha)
    print("ElasticNet RMSE on Training set :", rmse_cv(elasticNet, X, Y).mean())

    y_train_ela = elasticNet.predict(X)
    print('rmsle calculate by self:',rmsle(list(np.exp(Y)-1),list(np.exp(y_train_ela)-1)))

    # Plot residuals

    plt.scatter(y_train_ela, y_train_ela - Y, c="blue", marker="s", label="Training data")

    plt.title("Linear regression with ElasticNet regularization")
    plt.xlabel("Predicted values")
    plt.ylabel("Residuals")
    plt.legend(loc="upper left")
    plt.hlines(y=0, xmin=10.5, xmax=13.5, color="red")
    plt.show()
    # Plot predictions
    plt.scatter(Y, y_train_ela, c="blue", marker="s", label="Training data")

    plt.title("Linear regression with ElasticNet regularization")
    plt.xlabel("Predicted values")
    plt.ylabel("Real values")
    plt.legend(loc="upper left")
    plt.plot([10.5, 13.5], [10.5, 13.5], c="red")
    plt.show()
    return elasticNet

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
def GDBT_regression(X=train_split,Y=y):
    est = GradientBoostingRegressor(n_estimators=75,max_depth=3,learning_rate=0.1)
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=0)
    est.fit(X_train,Y_train)
    y_train_pred = est.predict(X_test)
    plt.scatter(y_train_pred,y_train_pred - Y_test,c = 'blue',marker='s', label='error on training data')

    plt.title("Linear regression with  GDBT")
    plt.xlabel("Predicted values")
    plt.ylabel("Residuals")
    plt.legend(loc="upper left")
    plt.hlines(y=0, xmin=10.5, xmax=13.5, color="red")
    plt.show()
    # Plot predictions
    plt.scatter(Y_test, y_train_pred, c="blue", marker="s", label="Training data")

    plt.title("Linear regression with  GDBT")
    plt.xlabel("Predicted values")
    plt.ylabel("Real values")
    plt.legend(loc="upper left")
    plt.plot([10.5, 13.5], [10.5, 13.5], c="red")
    plt.show()
    print('rmse value:',rmsle(Y_test,y_train_pred))

    return est


# linear_regression()
# ridge_regression()
# Lasso_regression()
#model = Elasticnet_regression()
# '''
#         predict final result
#  '''
#
#
# coefs,lasso = Lasso_regression()
# selected_features = coefs[coefs['value'] != 0].index.values
# train_new = train_split[selected_features]

model = GDBT_regression()
# pre_result = model.predict(test)
# pre_result = np.exp(pre_result) - 1
#
# df = pd.DataFrame(np.round(pre_result,2), index=list(range(1461, 1461 + pre_result.shape[0])), columns=['SalePrice'])
#
# print('length of pred result:', df.shape, '\n', df)
# df.to_csv('../../submission1219.csv',index_label='Id')



