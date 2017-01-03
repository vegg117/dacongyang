# coding=utf-8
# asdfasd


import pandas as pd
import numpy as np

from pandas import Series, DataFrame

train_data = pd.read_table("/data/risk_predict/train/user_info_train.txt", sep=',',
                           names=['id','sex','job','edu','marry','res'])

test_data = pd.read_table("/data/risk_predict/test/user_info_test.txt", sep=',',
                           names=['id','sex','job','edu','marry','res'])
print train_data.columns

data_train_overdue = pd.read_table("/data/risk_predict/train/overdue_train.txt", sep=',',
                                   names=['id', 'y'])
print data_train_overdue.columns


print "用户性别缺省的个数为:%s" % len(train_data[train_data.sex == 0])
# 用户性别缺省数为1669

exit()
def deal_user_table(df):
    print "++++++++++++++++++"
    print df.columns
    print "--------------------"
    dummies_sex = pd.get_dummies(df['sex'], prefix='sex')
    dummies_job = pd.get_dummies(df['job'], prefix='job')
    dummies_edu = pd.get_dummies(df['edu'], prefix='edu')
    dummies_marry = pd.get_dummies(df['marry'], prefix='marry')
    dummies_res = pd.get_dummies(df['res'], prefix='res')
    df = pd.concat([df, dummies_sex, dummies_job, dummies_edu, dummies_marry, dummies_res], axis=1)
    df.drop(['sex', 'job', 'edu', 'marry', 'res'], axis=1, inplace=True)
    return df

train_data = deal_user_table(train_data)
test_data = deal_user_table(test_data)

print 'ooooooooooooooooo'
print train_data.shape
print test_data.shape
print 'oooooooooooooooooo'

train_data = pd.merge(train_data, data_train_overdue)

y = train_data['y']
# train_data.drop(['y', 'id'], axis=1, inplace=True)
t_train = train_data.filter(regex='sex_.*|job_.*|edu_.*|marry_.*|res_.*')
#test_data.drop(['id'], axis=1, inplace=True)
t_test = test_data.filter(regex='sex_.*|job_.*|edu_.*|marry_.*|res_.*')
x = t_train


print 'ooooooooooooooooo'
print t_train.shape
print t_test.shape
print 'oooooooooooooooooo'


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
