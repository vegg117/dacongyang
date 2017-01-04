# coding=utf-8

import time
import pandas as pd
import numpy as np
from pandas import Series, DataFrame


df_train_user = pd.read_table("../data/risk_predict/train/user_info_train.txt", sep=',',
                           names=['uid','sex','job','edu','marry','res'])
print df_train_user.columns


# 数据检查
print "\n检查是否存在空值"
# 统计用户表用户性别缺省值个数
print "用户性别缺省的个数为:%s" % len(df_train_user[df_train_user.sex == 0]) # 用户性别缺省数为1669


# 用户表特征处理
def deal_user_feature(df):
    """
    对除uid外所有字段进行one-hot处理
    :param df:
    :return:
    """
    dummies_sex = pd.get_dummies(df['sex'], prefix='sex')
    dummies_job = pd.get_dummies(df['job'], prefix='job')
    dummies_edu = pd.get_dummies(df['edu'], prefix='edu')
    dummies_marry = pd.get_dummies(df['marry'], prefix='marry')
    dummies_res = pd.get_dummies(df['res'], prefix='res')
    df = pd.concat([df, dummies_sex, dummies_job, dummies_edu, dummies_marry, dummies_res], axis=1)
    df.drop(['sex', 'job', 'edu', 'marry', 'res'], axis=1, inplace=True)
    return df


def 