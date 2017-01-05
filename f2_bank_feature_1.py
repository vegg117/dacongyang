# coding=utf-8

import pandas as pd
import numpy as np
from pandas import Series, DataFrame
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

import f1_user_feature_1 as user
import function_tool as tool

en_remainMoney = False
en_transCount = True
# 训练集特征
n_fea = pd.DataFrame()
# 测试集特征
t_fea = pd.DataFrame()

en_navi = True
en_navi = False

if(en_navi):
    df_train = pd.read_table("../data/risk_predict/train/small.bank_detail_train.csv", sep=',', names=['uid', 'time', 'type', 'money', 'isPayout'])
    df_test = pd.read_table("../data/risk_predict/test/small.bank_detail_test.csv", sep=',', names=['uid', 'time', 'type', 'money', 'isPayout'])
else:
    df_train = pd.read_table("../data/risk_predict/train/bank_detail_train.txt", sep=',',
                              names=['uid', 'time', 'type', 'money', 'isPayout'])
    df_test = pd.read_table("../data/risk_predict/test/bank_detail_test.txt", sep=',',
                           names=['uid', 'time', 'type', 'money', 'isPayout'])




def get_info():
    print "\n检查是否存在空值"
    print df_train.info(null_counts=True)
    print "时间戳缺省的个数为:%s" % len(df_train[df_train.time == 0])



n_userid = df_train['uid'].unique()
n_fea = pd.DataFrame({'uid':n_userid})
print "train bank has",df_train.shape[0],"record"
print "train bank has",len(n_userid),"user"

# t_userid = df_test['uid'].unique()
t_userid = user.get_test_userid()
t_fea = pd.DataFrame({'uid':t_userid})
print "test bank has", df_test.shape[0] , "record"
print "test bank has", len(df_test['uid'].unique()) , "user"
print "test bank 使用test user表的userid"
# exit()
# print n_fea.head()
# print t_fea.head()
# exit()

def get_payCount_fea():
    return
def get_inCount_fea():
    return
def get_remainMoney_fea():
    '''

    :return:
    '''

    return

def deal_transCount(df, uid):
    '''
    交易次数特征
    :return:
    '''
    idlen = len(uid)
    i = 0
    transCount = pd.Series([])
    while (i < idlen):
        res = df[df.uid == uid[i]]
        #print "userid"
        #print userid[i]
        #print i
        record = res.shape[0]
        if (record > 0):
            # print uid[i]
            # print record
            transCount[i] = record
        i = i + 1
    # print transCount.shape
    # print transCount.count()
    # print uid
    # print "=============="
    # print transCount

    res = pd.DataFrame({'uid':uid, 'transCount': transCount})
    # print res.head(3)
    # print res

    return res

def get_transCount_fea(df_train, df_test, n_uid, t_uid):
    '''
    交易次数特征
    :return:
    '''
    # print df_train.shape
    # print df_test.shape
    # print len(n_uid)
    # print len(t_uid)
    # print df_test
    # print t_uid
    # exit()

    # print "start get_transCount_fea, this is a long long long..."

    n_trans = deal_transCount(df_train, n_uid)
    # print "train bank 缺失值个数是", len(n_trans[n_trans.transCount.isnull()])


    print "train end, test start..."
    t_trans = deal_transCount(df_test, t_uid)
    print "test bank 缺失值个数是", len(t_trans[t_trans.transCount.isnull()])

    # n_trans = t_trans
    print n_trans.shape
    print t_trans.shape
    print "transCount特征的空值使用0填充"
    n_trans.fillna(0, inplace=True)
    t_trans.fillna(0, inplace=True)
    print "train bank 缺失值个数是", len(n_trans[n_trans.transCount.isnull()])
    print "test bank 缺失值个数是", len(t_trans[t_trans.transCount.isnull()])

    print n_trans.head()
    print t_trans.head()

    # exit()
    # 归一化
    print "standard scaler...."
    ss = StandardScaler()
    n_tc = ss.fit_transform(n_trans['transCount'])
    t_tc = ss.transform(t_trans['transCount'])
    print n_tc
    print t_tc

    # n_tc = (n_tc+1)/2
    # t_tc = (t_tc+1)/2
    n_trans.transCount = n_tc
    t_trans.transCount = t_tc

    print n_trans.head()
    print t_trans.head()

    print "transcation count 归一化后负值个数分别为", len(n_trans[n_trans.transCount < 0]), len(t_trans[t_trans.transCount < 0])

    return n_trans, t_trans


def get_xxx(df_train, df_test):
    '''
    挖掘xxx特征，
    :param df_train:
    :param df_test:
    :return: train, test
    '''

    # 提取相关数据
    df_train = df_train['xxx', 'yyy']

    # 处理训练集特征
    mean = df_train.columns.mean()
    train = df_train[1]

    # 处理测试集特征
    test = df_test[1] - mean

    return train, test

if en_transCount :
    n_trans, t_trans = get_transCount_fea(df_train, df_test, n_userid, t_userid)
    # 其他调试
    #print "\nn_trans"
    #print n_trans
    #print "\nn_fea"
    #print n_fea
    n_fea = pd.merge(n_fea, n_trans, on='uid')
    t_fea = pd.merge(t_fea, t_trans, on='uid')
    print n_fea
    print t_fea

def get_fea():

    print n_fea
    print t_fea

    return n_fea, t_fea

# get_fea()

