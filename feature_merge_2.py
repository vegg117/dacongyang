# coding=utf-8

import time
import pandas as pd
import numpy as np
from pandas import Series, DataFrame

import f1_user_feature_1 as user
import f2_bank_feature_1 as bank

'''
    feature_merge
     1、获取各表的特征处理结果。对训练集测试集同样操作，调用分feature函数，传入分feature的df，获得处理后的数据
     2、拼接。（1）将训练集的特征和结果y按uid拼接;(2)将test的特征进行拼接
     3、分离train的输入输出:x_train,y_train
     4、去掉x_test、x_train的uid，并将test的uid保存到y_id
'''

start = time.clock()

df_train_user = pd.read_table("../data/risk_predict/train/user_info_train.txt", sep=',',
                           names=['uid','sex','job','edu','marry','res'])

print df_train_user.columns

df_train_bank = pd.read_table("../data/risk_predict/train/bank_detail_train.txt", sep=',',
                              names=['uid', 'time', 'type', 'money', 'isPayout'])
print df_train_bank.columns


df_test_user = pd.read_table("../data/risk_predict/test/user_info_test.txt", sep=',',
                           names=['uid','sex','job','edu','marry','res'])

df_test_bank = pd.read_table("../data/risk_predict/test/bank_detail_test.txt", sep=',',
                           names=['uid', 'time', 'type', 'money', 'isPayout'])

df__train_overdue = pd.read_table("../data/risk_predict/train/overdue_train.txt", sep=',',
                                   names=['uid', 'y'])
print df__train_overdue.columns

#  1、对训练集测试集同样操作，调用子feature函数，传入子feature的df，获得处理后的数据
def deal_all_features(fuser, fbank):
    """
    处理所有的特征，此函数为方便训练集与测试集做同样的处理
    :param user:
    :param bank:
    :return:
    """
    fuser = user.deal_user_feature(fuser)
    fbank = bank.deal_bank_feature(fbank)
    return fuser, fbank


df_train_user, df_train_bank = deal_all_features(df_train_user, df_train_bank)
df_test_user, df_test_bank = deal_all_features(df_test_user, df_test_bank)

print "\n\n\n\n\n\n\n2、拼接..."
print "拼接前user bank overdue .shape"
print df_train_user.shape
print df_train_bank.shape
print df__train_overdue.shape
print df_train_user.head(2)
print df_train_bank.head(2)

# 2、拼接。（1）将训练集的特征和结果y按uid拼接;(2)将test的特征进行拼接
# (1)拼接 以uid为键拼接三个表
print "\ndf_train_user的特征个数为：%s" % len(df_train_user.columns)
print "\ndf_train_bank的特征个数为：%s" % len(df_train_bank.columns)
df_train_data = pd.merge(df_train_user, df_train_bank)
print df_train_data.columns
print "拼接后的shape："
print df_train_data.shape

print "\ndf_train_data的特征个数为：%s" % len(df_train_data.columns)
df_train_data = pd.merge(df_train_data, df__train_overdue)
print df_train_data.columns
print "\ndf_train_data的特征个数为：%s" % len(df_train_data.columns)
print "拼接后的shape："
print df_train_data.shape
print df_train_data.head()
# (2)拼接 以uid为键拼接两个表4、去掉x_test、x_train的uid，并将test的uid保存到y_id
df_test_data = pd.merge(df_test_user, df_test_bank)
print df_train_data.head()


# 3、分离train的输入输出:x_train,y_train
# 分割训练集的输入和输出(x,y)
y_train = df_train_data['y']
df_train_data.drop(['y'], axis=1, inplace=True)

# 4、去掉x_test、x_train的uid，并将test的uid保存到y_id
df_train_data.drop(['uid'], axis=1, inplace=True)
x_train = df_train_data
y_id = df_test_data['uid']
print "\ny_id:\n%s" % y_id.head()
df_test_data.drop(['uid'], axis=1, inplace=True)
x_test = df_test_data

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
    return x_train, y_train, x_test, y_id