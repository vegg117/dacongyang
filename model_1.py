# coding=utf-8

import time
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
from sklearn import preprocessing
from xgboost import XGBClassifier
from xgboost import XGBRegressor
import feature_merge_2 as feature


start = time.clock()

# 数据获取
x_train, y_train, x_test, y_id = feature.get_data()

end = time.clock()
print "run time: %f s" % (end - start)
print "start model..."

# 模型
rfr = RandomForestRegressor(n_estimators=10)
rfr.fit(x_train, y_train)

print "x_train"
print x_train.head()
print "y_train"
print y_train.head()
print "x_test"
print x_test.head()
# exit()
pre_y = rfr.predict(x_test)
print pre_y
print "一共有",len(pre_y),"条记录"
print
# exit()

result = pd.DataFrame({'auserid':y_id.as_matrix(), 'probability':pre_y.astype(np.float32)})
print result.head()
result.to_csv("../data/risk_predict/result/t0105_7.csv", index=False)

end = time.clock()
print "run time: %f s" % (end - start)
exit()



























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


rf = RandomForestRegressor(n_estimators = 100, n_jobs=-1).fit(x, y)
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
result = pd.DataFrame({'auserid':y_id['uid'].as_matrix(), 'probability':pre_y_embed.astype(np.float32)})
print sum(result['probability'].round())
print result.head()
result.to_csv("/data/risk_predict/result/0102_2.model_embed.csv", index=False)

exit()

