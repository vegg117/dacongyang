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

df_train_browse = pd.read_table("/data/risk_predict/train/browse_history_train.txt", sep=',',
                              names=['uid', 'time', 'behaviour', 'subBehId'])
print df_train_browse.columns


#df_train_overdue = pd.read_table("/data/risk_predict/train/overdue_train.txt", sep=',',
#                                   names=['id', 'y'])

print "bank info()"
print df_train_browse.info(null_counts=True)
print df_train_browse.head()

# 统计时间为空值的个数（时间戳为0表示空值）
print "时间戳缺省的个数为:%s" % len(df_train_browse[df_train_browse.time == 0])
# 时间戳缺省个数为0

# 浏览行为表特征处理 one-hot
def deal_bank_feature(df):
    dummies_type = pd.get_dummies(df['type'], prefix='type')
    dummies_isPayout = pd.get_dummies(df['isPayout'], prefix='isPayout')
    df = pd.concat([df, dummies_type, dummies_isPayout], axis=1)        #axis=1是按列连接（二维）
    df.drop(['type', 'isPayout'], axis=1, inplace=True)
    return df

# 对交易类型进行one-hot处理
df_train_bank_origin = df_train_browse
df_train_bank = deal_bank_feature(df_train_browse)
print "df_train_bank_origin.columns \n%s" % df_train_bank_origin.columns
print "df_train_bank.columns \n%s" % df_train_bank.columns
print "df_train_bank_origin.head(2) \n%s" % df_train_bank_origin.head(2)
print "df_train_bank.head(2) \n%s" % df_train_bank.head(2)

# 去掉属性uid
df_train_bank.drop(['uid'], axis=1, inplace=True)
print df_train_bank.columns
print df_train_bank_origin.columns

#



#print sum(data_train_overdue['overdue'].round())