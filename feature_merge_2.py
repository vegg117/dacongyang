# coding=utf-8

import time
import pandas as pd
import numpy as np
from pandas import Series, DataFrame

import f1_user_feature_1 as user
import f2_bank_feature_1 as bank
import f3_browse_feature_1 as browse
import superTools as tool

'''
    feature_merge
     1、获取各表的特征处理结果。
     2、拼接。
        将各表特征按uid拼接为一个表。分为train和test
        （1）将训练集的特征和结果y按uid拼接;
     3、分离train的输入输出:x_train,y_train
     4、去掉x_test、x_train的uid，并将test的uid保存到y_id
'''

start = time.clock()

df__train_overdue = pd.read_table("../data/risk_predict/train/overdue_train.txt", sep=',',
                                   names=['uid', 'y'])


df_train_user, df_test_user = user.get_fea()

print df_train_user.head(10)
df_train_bank, df_test_bank = bank.get_fea()

df_train_browse, df_test_browse = browse.get_fea()


# test
# print df_test_bank.shape
# print df_test_user.shape


#exit()



print "\n\n\n\n\n\n\n2、拼接..."
# print "拼接前user bank overdue .shape"
# print df_train_user.shape
# print df_train_bank.shape
# print df__train_overdue.shape
# print df_train_user.head(2)
# print df_train_bank.head(2)


# exit()
# 2、拼接。（1）将训练集的特征和结果y按uid拼接;(2)将test的特征进行拼接
# (1)拼接 以uid为键拼接三个表
print "\ndf_train_user的特征个数为：%s" % len(df_train_user.columns)
print "\ndf_train_bank的特征个数为：%s" % len(df_train_bank.columns)
df_train_data = pd.merge(df_train_user, df_train_bank, on='uid')
df_train_data = pd.merge(df_train_data, df_train_browse, on='uid')
print df_train_data.columns
#print "拼接后的shape："
# print df_train_data.shape
# print df_train_data.head(3)

print "\ndf_train_data的特征个数为：%s" % len(df_train_data.columns)
df_train_data = pd.merge(df_train_data, df__train_overdue, on='uid')
print df_train_data.columns
print "\ndf_train_data的特征个数为：%s" % len(df_train_data.columns)
print "拼接后的shape：" , df_train_data.shape
# print df_train_data.head(3)
# (2)拼接 以uid为键拼接两个表4、去掉x_test、x_train的uid，并将test的uid保存到y_id
df_test_data = pd.merge(df_test_user, df_test_bank, on='uid')
df_test_data = pd.merge(df_test_data, df_test_browse, on='uid')
# print df_train_data.head(3)

#exit()

# 3、分离train的输入输出:x_train,y_train
# 分割训练集的输入和输出(x,y)
y_train = df_train_data['y']
df_train_data.drop(['y'], axis=1, inplace=True)

# test
# print y_train.shape
# exit()

# 4、去掉x_test、x_train的uid，并将test的uid保存到y_id
df_train_data.drop(['uid'], axis=1, inplace=True)
x_train = df_train_data
y_id = df_test_data['uid']
# print "\ny_id:\n%s" % y_id.head()
df_test_data.drop(['uid'], axis=1, inplace=True)
x_test = df_test_data


# test
print df_train_data.shape
print "----------------"
print "x_train的列属性有：%s\n" % x_train.columns
print "x_test的列属性有：%s\n" % x_test.columns
print "x_train.shape:"
print x_train.shape
print "x_test.shape:"
print x_test.shape
print "y_train.shape:"
print y_train.shape
print "y_id.shape:"
print y_id.shape
print "x_train type: "
print type (x_train)
print "x_test type:"
print type (x_test)
print "y_train type:"
print type (y_train)
print "y_id type:"
print type (y_id)
print "----------------------\n\n"



def get_data():
    """
    :return:
        x_train type: <class 'pandas.core.frame.DataFrame'>
        x_test type: <class 'pandas.core.frame.DataFrame'>
        y_train type: <class 'pandas.core.series.Series'>
        y_id type: <class 'pandas.core.series.Series'
    """

    print '--------------------------------------'
    print x_train
    print x_test
    print y_train
    print y_id

    return x_train, y_train, x_test, y_id

