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
from xgboost import XGBRegressor
from xgboost import XGBClassifier

df_train_data = pd.read_table("/data/risk_predict/train/user_info_train.txt", sep=',',
                           names=['id','sex','job','edu','marry','res'])

test_data = pd.read_table("/data/risk_predict/test/user_info_test.txt", sep=',',
                           names=['id','sex','job','edu','marry','res'])
print df_train_data.columns

data_train_overdue = pd.read_table("/data/risk_predict/train/overdue_train.txt", sep=',',
                                   names=['id', 'y'])
print data_train_overdue.columns

# 用户表特征处理 one-hot
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
t_test = test_data.filter(regex='sex_.*|job_.*|edu_.*|marry_.*|res_.*')
x = t_train.values


print 'ooooooooooooooooo'
print t_train.shape
print t_test.shape
print 'oooooooooooooooooo'

# x y 分别是训练集的输入和输出
# t_test 是验证集的输入  test是未经处理的，保留用户id的DF
# t_train与df_tarin_data区别：t_train去掉了用户id
# t_train与x相同


print "模型开始"
# 模型
# 将已有的训练集数据分为训练集和测试集
print "len_x:%s" % len(x)
cv = cross_validation.ShuffleSplit(len(x), n_iter=3, test_size=0.2, random_state=0)


print "神器/XGBRegressor"
for train, test in cv:
    svc = XGBClassifier().fit(x[train], y[train])
    print("train score: {0:.3f}, test score: {1:.3f}\n".format(
        svc.score(x[train], y[train]), svc.score(x[test], y[test])))

exit()

# 领回归
print "领回归"
for train, test in cv:
    svc = linear_model.Ridge().fit(x[train], y[train])
    print("train score: {0:.3f}, test score: {1:.3f}\n".format(
    svc.score(x[train], y[train]), svc.score(x[test], y[test])))


# 逻辑回归
print "逻辑回归"
for train, test in cv:
    svc = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6).fit(x, y)
    print("train score: {0:.3f}, test score: {1:.3f}\n".format(
    svc.score(x[train], y[train]), svc.score(x[test], y[test])))


'''
print "支持向量回归/SVR(kernel='rbf',C=10,gamma=.001)"
for train, test in cv:
    svc = svm.SVR(kernel='rbf', C=10, gamma=.001).fit(x[train], y[train])
    print("train score: {0:.3f}, test score: {1:.3f}\n".format(
        svc.score(x[train], y[train]), svc.score(x[test], y[test])))
'''

'''
print "随机森林回归/Random Forest(n_estimators = 100)"
for train, test in cv:
    svc = RandomForestRegressor(n_estimators = 10).fit(x[train], y[train])
    print("train score: {0:.3f}, test score: {1:.3f}\n".format(
        svc.score(x[train], y[train]), svc.score(x[test], y[test])))
'''

# 开始参数调优
print "开始参数调优"
x_train, x_test, y_train, y_test = cross_validation.train_test_split(
    x, y, test_size=0.2, random_state=0)
tuned_parameters = [{'n_estimators':[10,100,500, 1000]}]
scores=['r2']
for score in scores:
    print score
    clf = GridSearchCV(RandomForestRegressor(), tuned_parameters, cv=5, scoring=score)
    clf.fit(x_train, y_train)
    print "find the best paragram"
    print ""
    print clf.best_estimator_
    print
    print "得分分别是"
    print
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))
    print ""


print "exit..."
exit()



# randomforest
from sklearn.ensemble import  RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=100)
rfr.fit(x, y)
pre_y = rfr.predict(t_test)
print "(((((((((((((((((((((((((((((("
print pre_y


result = pd.DataFrame({'id':test_data['id'].as_matrix(), 'overdue':pre_y.astype(np.float32)})
print result.head()
result.to_csv("/data/risk_predict/result/logistic_regression_predictions.csv", index=False)


exit()

print "-------------------"
from sklearn import linear_model

clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(x, y)

print "clf: %s" % (clf)

predictions = clf.predict(t_test)

print "predict: %s " % predictions


result = pd.DataFrame({'id':test_data['id'].as_matrix(), 'overdue':predictions.astype(np.int32)})
print result.head()
#result.to_csv("logistic_regression_predictions.csv", index=False)


from sklearn.metrics import classification_report

# print classification_report(y, predictions)



# train_x = x_mat[:]