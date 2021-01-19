"""


@author: mobiiina199
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn.linear_model import LinearRegression, BayesianRidge
import math
import time
from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error


train = pd.read_csv('E:/data/data science/data course/Final  DS Project/dataset/train-mobina_final.csv')
test = pd.read_csv('.../test-mobiiina199.csv')

y = np.log(train.SalePrice)
X = train.drop(['SalePrice'], axis=1)

# scale
'''scalar = RobustScaler()
X = scalar.fit_transform(X)'''

# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# average SalePrice in train and test
print('mean SalePrice in train : {0:.3f}'.format(np.mean(y_train)))
print('mean SalePrice in test : {0:.3f}'.format(np.mean(y_test)))

from sklearn.dummy import DummyRegressor

# baseline model
model_dummy = DummyRegressor(strategy='mean')
model_dummy.fit(X_train, y_train)
print('score for baseline model: {0:.2f}'.format(model_dummy.score(X_test, y_test)))


def find_best_model_using_gridsearchcv(X, y):
    algos = {
            'linear_regression': {
                    'model': LinearRegression(),
                    'params': {
                            'normalize': [True, False]
                            }
                    },
            'lasso': {
                    'model': Lasso(),
                    'params': {
                            'alpha': [1, 2, 3, 4, 5],
                            'selection': ['random', 'cyclic']
                            }
                    },
            'ridge': {
                    'model': Ridge(),
                    'params': {
                            'alpha': [100, 200, 300],
                            'normalize': [True, False]
                            }
                    },
            'decision_tree': {
                    'model': DecisionTreeRegressor(),
                    'params': {
                            'criterion': ['mse', 'friedman_mse', 'mae'],
                            'splitter': ['best', 'random'],
                            'max_depth': [4, 8, 9, 10],
                            'min_samples_split': [2, 5, 8, 11],
                            'max_features': ['auto', 'sqrt', 'log2'],
                            'ccp_alpha': [0, 0.05, 0.1]
                            }
                    },
            'random_forest': {
                    'model': RandomForestRegressor(),
                    'params': {
                            'n_estimators': [100, 300, 500, 600],
                            'criterion': ['mse', 'mae'],
                            'max_depth': [9, 12, 15, 18],
                            'max_features': ['auto', 'sqrt', 'log2']
                            }
                    },
            'xgbr': {
                    'model': xgb.XGBRegressor(),
                    'params': {
                            'booster': ['gbtree', 'gblinear'],
                            'learning_rate': [0.001, 0.01, 0.1, 0.2],
                            'n_estimators': [100, 200, 250, 300]
                            }
                    }
            }

    scores = []

    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs = GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X_train, y_train)
        scores.append({
               'model': algo_name,
               'best_score': gs.best_score_,
               'best_params': gs.best_params_
            })
    return pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])


find_best_model_using_gridsearchcv(X, y)




# random forest
rf = RandomForestRegressor(criterion='mae', max_depth=15, max_features='auto', n_estimators=600)
rf.fit(X_train, y_train)
rf.score(X_test, y_test)
rfpredict = rf.predict(X_test)
mean_absolute_error(y_test, rfpredict)

# xgboost
xgb_reg = xgb.XGBRegressor(booster='gbtree', learning_rate=0.1, n_estimators=200)
xgb_reg.fit(X_train, y_train)
ypred = xgb_reg.predict(X_test)
mean_absolute_error(y_test, ypred)

''''''
test_X = test.values.astype('float')
predictions = xgb_reg.predict(test_X)

predSalePrice = pd.DataFrame(np.exp(predictions), columns=['predSalePrice'])
test['predSalePrice'] = predSalePrice

test.to_csv('E:/data/data science/data course/Final  DS Project/dataset/test-mobina-final.csv')
# dimention redaction with PCA
pca = PCA()
blackbox_model = Pipeline([('pca', pca), ('rf', rf)])
blackbox_model.fit(X_train, y_train)
blackbox_model.score(X_test, y_test)

# features importance
FeatureImp_xgb = pd.DataFrame({'index': X_train.columns, 'feature_importance': xgbreg.feature_importances_})
FeatureImp_xgb.sort_values(by='feature_importance', ascending=False, inplace=True)
f, ax = plt.subplots(1, 1, figsize=[12, 9])
sns.barplot(x='feature_importance', y='index', data=FeatureImp_xgb.iloc[:15, ], ax=ax)


FeatureImp_rf = pd.DataFrame({'index': X_train.columns, 'feature_importance': rf.feature_importances_})
FeatureImp_rf.sort_values(by='feature_importance', ascending=False, inplace=True)
f, ax = plt.subplots(1, 1, figsize=[9, 7])
sns.barplot(x='feature_importance', y='index', data=FeatureImp_rf.iloc[:15, ], ax=ax)

import pickle

pickle_file_name = 'E:/data/data science/data course/Final  DS Project/house_price_model.PKl'
with open(pickle_file_name, 'wb') as file: 
    pickle.dump(rf, file)   
    
pickle.load(open('E:/data/data science/data course/Final  DS Project/house_price_model.PKl','rb'))
