# coding=utf-8

import pandas as pd
import numpy as np
from pandas import Series, DataFrame
from sklearn import preprocessing


df_train_bank = pd.read_table("../data/risk_predict/train/bank_detail_train.txt", sep=',',
                              names=['uid', 'time', 'type', 'money', 'isPayout'])
print df_train_bank.columns


# 数据清洗
print "\n检查是否存在空值"
print df_train_bank.info(null_counts=True)
print "时间戳缺省的个数为:%s" % len(df_train_bank[df_train_bank.time == 0])
    # 时间戳缺省个数为38773    数量大，考虑使用其作为标签，使用其他特征估算



# 流水表特征处理
def deal_bank_feature(df):
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
