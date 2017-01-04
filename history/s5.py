# coding=utf-8

import pandas as pd
import numpy as np
from pandas import Series, DataFrame
from sklearn import linear_model
from sklearn import cross_validation
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.learning_curve import learning_curve
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import explained_variance_score
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

data_train_overdue = pd.read_table("/data/risk_predict/result/logistic_regression_predictions.csv", sep=',')
print data_train_overdue.columns

print sum(data_train_overdue['overdue'].round())