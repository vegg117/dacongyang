# coding=utf-8

import time
import pandas as pd
import numpy as np
from pandas import Series, DataFrame



en_navi = True
en_navi = False

if(en_navi):
    df_train = pd.read_table("../data/risk_predict/train/small.user_info_train.csv", sep=',',
                             names=['uid', 'sex', 'job', 'edu', 'marry', 'res'])

    df_test = pd.read_table("../data/risk_predict/test/small.user_info_test.csv", sep=',',
                            names=['uid', 'sex', 'job', 'edu', 'marry', 'res'])
else:
    df_train = pd.read_table("../data/risk_predict/train/user_info_train.txt", sep=',',
                           names=['uid','sex','job','edu','marry','res'])

    df_test = pd.read_table("../data/risk_predict/test/user_info_test.txt", sep=',',
                           names=['uid','sex','job','edu','marry','res'])


n_userid = df_train['uid'].unique()
print "user train一共有", len(n_userid) , "个用户"
t_userid = df_test['uid'].unique()
n_fea = pd.DataFrame({'uid':n_userid})
t_fea = pd.DataFrame({'uid':t_userid})

print "user test has", df_test.shape[0], "条记录"
print "user test has", len(t_userid), "record"
# exit()
def get_info():
    print "查看用户id是否唯一："
    print df_train.uid.is_unique
#    print df_train_user.sex.is_unique
    print "\n检查是否存在空值"
    print df_train.info(null_counts=True)
    # 统计用户表用户性别缺省值个数
    print "用户性别缺省的个数为:%s" % len(df_train[df_train.sex == 0]) # 用户性别缺省数为1669

#get_info()


def deal_fea(df):
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

def get_fea():
    n_all = deal_fea(df_train)
    t_all = deal_fea(df_test)
    n_fea1 = pd.merge(n_fea, n_all, on='uid')
    t_fea1 = pd.merge(t_fea, t_all, on='uid')
    # print n_fea1.shape
    # print t_fea1.shape
    # print t_all.shape
    # print t_fea.shape
    return n_fea1, t_fea1


# a, b =get_fea()
# print b.shape
# print a.shape




def get_test_userid():
    return df_test['uid']
