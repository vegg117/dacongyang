# code=utf-8


import pandas as pd
import numpy as np

from pandas import Series, DataFrame

def test():
    print "test"

data_train = pd.read_table("/data/risk_predict/train/user_info_train.txt", sep=',')
print data_train.columns

exit()
data_train_overdue = pd.read_table("/data/risk_predict/train/overdue_train.txt", sep=',')
print data_train_overdue.columns

data_train.info()

new_train = pd.merge(data_train, data_train_overdue)
print new_train.columns

print new_train.head()

new_train.info()

dummies_sex = pd.get_dummies(new_train['sex'], prefix='sex')

dummies_job = pd.get_dummies(new_train['job'], prefix='job')

dummies_edu = pd.get_dummies(new_train['edu'], prefix='edu')

dummies_marry = pd.get_dummies(new_train['marry'], prefix='marry')

dummies_res = pd.get_dummies(new_train['res'], prefix='res')

df = pd.concat([new_train, dummies_sex, dummies_job, dummies_edu, dummies_marry, dummies_res], axis=1)

df.drop(['sex', 'job', 'edu', 'marry', 'res'], axis=1, inplace=True)
print '-----------------'
print df.columns
print df.head()
print

y = new_train['y']
df.drop(['y', 'id'], axis=1, inplace=True)
x = df
print x.head()
print y.head()

print x.shape[0]
len = x.shape[0]


x_mat = x.as_matrix()
print x_mat[1]
y_mat = y.as_matrix()
print y_mat[1]

ratio = 0.7
train_len = int(len*ratio)
print "train_len:%s" % train_len
x_train = x_mat[:train_len, :]
y_train = y_mat[:train_len]

x_test = x_mat[train_len:, :]
y_test = y_mat[train_len:]

print x_train[3:3, :]

print "-------------------"
from sklearn import linear_model

clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(x_train, y_train)

print "clf: %s" % (clf)


predictions = clf.predict(x_test)


print "predict: %s " % predictions


from sklearn.metrics import classification_report
print classification_report(y_test, predictions)



#train_x = x_mat[:]