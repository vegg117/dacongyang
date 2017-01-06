# coding=utf-8


import pandas as pd
import numpy as np
from pandas import Series, DataFrame
from sklearn import preprocessing

import f1_user_feature_1 as user


df_train = pd.read_table("../data/risk_predict/train/browse_history_train.txt", sep=',',
                              names=['uid', 'time', 'behaviour', 'subBehId'])


# 数据清洗
def get_info():
    uNum = len(df_train.uid.unique())
    print "------------------------\nbrowse_history_train表信息\n"
    # print "describe：\n", df_train.describe()
    print "\ninfo:\n", df_train.info(null_counts=True)
    print "\ncount:\n", df_train.count()
    print "\nshape:" , df_train.shape
    print "该表存在的用户数：", uNum
    print "一个用户的记录数是否唯一：", df_train.uid.is_unique
    print "没有该表记录的用户数：", (len(user.get_train_userid())-uNum)
    print "时间戳缺省的个数为:%s" % len(df_train[df_train.time == 0])
    print "用户浏览行为最大最小值：", max(df_train.behaviour)
    print "用户浏览行为最大值：", max(df_train.behaviour) - min(df_train.behaviour)
    print "用户浏览子行为最大值：", max(df_train.subBehId)
    print "用户浏览子行为最大最小值差：", max(df_train.subBehId) - min(df_train.subBehId)

    '''
        观察发现：
        train：
            总用户数：55596, 该表用户数：47330，有8266个用户没有记录
            一个用户有多条记录，也不是所有的用户均有记录
            时间没有缺省
            用户浏览行为最大最小差215, 最大值为215
            浏览子行为编号差10，最大值为11
    '''
get_info()
exit()

# 特征处理
def deal_bank_feature(df):
    '''
    对时间戳进行了删除

    :param df:
    :return:
    '''
    dummies_type = pd.get_dummies(df['type'], prefix='type')
    dummies_isPayout = pd.get_dummies(df['isPayout'], prefix='isPayout')
    df = pd.concat([df, dummies_type, dummies_isPayout], axis=1)        #axis=1是按列连接（二维）
    df.drop(['type', 'isPayout'], axis=1, inplace=True)
    #对金额进行归一化处理
    # 这个是将money列化到[-1,1]之间，如何到[0,1]之间？
    scaler = preprocessing.StandardScaler()
    money_scale_param = scaler.fit(df_train_bank['money'])
    df_train_bank['money_scaled'] = scaler.fit_transform(df_train_bank['money'], money_scale_param)
    # 去掉时间戳列
    df.drop(['time'], axis = 1, inplace = True)
    return df
