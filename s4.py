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

df_train_data = pd.read_table("/data/risk_predict/train/user_info_train.txt", sep=',',
                           names=['id','sex','job','edu','marry','res'])

test_data = pd.read_table("/data/risk_predict/test/user_info_test.txt", sep=',',
                           names=['id','sex','job','edu','marry','res'])
print df_train_data.columns

data_train_overdue = pd.read_table("/data/risk_predict/train/overdue_train.txt", sep=',',
                                   names=['id', 'y'])
print data_train_overdue.columns

# 特征处理 one-hot
def deal_user_table(df):
    dummies_sex = pd.get_dummies(df['sex'], prefix='sex')
    dummies_job = pd.get_dummies(df['job'], prefix='job')
    dummies_edu = pd.get_dummies(df['edu'], prefix='edu')
    dummies_marry = pd.get_dummies(df['marry'], prefix='marry')
    dummies_res = pd.get_dummies(df['res'], prefix='res')
    df = pd.concat([df, dummies_sex, dummies_job, dummies_edu, dummies_marry, dummies_res], axis=1)
    df.drop(['sex', 'job', 'edu', 'marry', 'res'], axis=1, inplace=True)
    return df

df_train_data = deal_user_table(df_train_data)
test_data = deal_user_table(test_data)


df_train_data = pd.merge(df_train_data, data_train_overdue)

y = df_train_data['y'].values
# train_data.drop(['y', 'id'], axis=1, inplace=True)
t_train = df_train_data.filter(regex='sex_.*|job_.*|edu_.*|marry_.*|res_.*', axis = 1)
#test_data.drop(['id'], axis=1, inplace=True)
t_test = test_data.filter(regex='sex_.*|job_.*|edu_.*|marry_.*|res_.*', axis = 1)
x = t_train.values
train_len = t_train.shape[0]
test_len = t_test.shape[0]

print type (test_data)
print type (test_data['id'])
print type (test_data['id'].as_matrix())


exit()

print 'ooooooooooooooooo'
print t_train.shape
print t_test.shape
print 'oooooooooooooooooo'

# x y 分别是训练集的输入和输出
# t_test 是验证集的输入  test是未经处理的，保留用户id的DF
# t_train与df_tarin_data区别：t_train去掉了用户id
# t_train与x相同

from xgboost import XGBRegressor

print np.sum(y)


lmr = linear_model.Ridge().fit(x, y)
pre_y_lmr = lmr.predict(x)
pre_y_lmr.shape = (train_len, 1)
print "pre_y_lmr:%s, %s, sum:%s" % (pre_y_lmr, pre_y_lmr.shape, sum(pre_y_lmr.round()))


lmlr = linear_model.SGDRegressor(penalty='l2').fit(x, y)
pre_y_lmlr = lmlr.predict(x)
pre_y_lmlr.shape = (train_len, 1)
print "pre_y_lmlr:%s" % pre_y_lmlr

print "pre_y_lmlr:%s, %s, sum:%s" % (pre_y_lmlr, pre_y_lmlr.shape, sum(pre_y_lmlr.round()))


rf = RandomForestRegressor(n_estimators = 100).fit(x, y)
pre_y_rf = rf.predict(x)
pre_y_rf.shape = (train_len, 1)
print "pre_y_rf:%s, %s, sum:%s" % (pre_y_rf, pre_y_rf.shape, sum(pre_y_rf.round()))

xgb = XGBClassifier().fit(x, y)
pre_y_xgb = xgb.predict(x)
pre_y_xgb.shape = (train_len, 1)
print "pre_y_xgb:%s, %s, sum:%s" % (pre_y_xgb, pre_y_xgb.shape, sum(pre_y_xgb.round()))


x = np.concatenate((pre_y_lmr, pre_y_lmlr, pre_y_rf), axis=1)
print x[:3, :]
print "x, y shape: %s, %s" % (x.shape, y.shape)

lmlr_embed = RandomForestRegressor(n_estimators = 100).fit(x, y)


# 生成答案
pre_y_lmr = lmr.predict(t_test.values)
pre_y_lmr.shape = (test_len, 1)
pre_y_lmlr = lmlr.predict(t_test.values)
pre_y_lmlr.shape = (test_len, 1)
pre_y_rf = rf.predict(t_test.values)
pre_y_rf.shape = (test_len, 1)
x = np.concatenate((pre_y_lmr, pre_y_lmlr, pre_y_rf), axis=1)
pre_y_embed = lmlr_embed.predict(x)
result = pd.DataFrame({'auserid':test_data['id'].as_matrix(), 'probability':pre_y_embed.astype(np.float32)})
print sum(result['probability'].round())
print result.head()
result.to_csv("/data/risk_predict/result/0102_2.model_embed.csv", index=False)

exit()

