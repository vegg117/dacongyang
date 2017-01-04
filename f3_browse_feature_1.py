# coding=utf-8

import pandas as pd
import numpy as np
from pandas import Series, DataFrame
from sklearn import preprocessing


df_train_browse = pd.read_table("../data/risk_predict/train/browse_history_train.txt", sep=',',
                              names=['uid', 'time', 'behaviour', 'subBehId'])


# 数据清洗
def get_info():
    print "browse_history_train表信息："
    print df_train_browse.describe()
    print
    print df_train_browse.count()
    print
    print "shape:"
    print df_train_browse.shape
    print df_train_browse.info(null_counts=True)
    print
    print "用户id是否唯一："
    print df_train_browse.uid.is_unique
    print
    print "时间戳缺省的个数为:%s" % len(df_train_browse[df_train_browse.time == 0])
    print
    print "用户浏览行为最大最小值差："
    print max(df_train_browse.behaviour) - min(df_train_browse.behaviour)
    print
    print "用户浏览行为最大最小值差："
    print max(df_train_browse.subBehId) - min(df_train_browse.subBehId)

    '''
        观察发现：
            一个用户会有多条记录
            时间没有缺省
            用户浏览行为最大最小差215
            浏览子行为编号差10
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
