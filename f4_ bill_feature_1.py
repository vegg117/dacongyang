# coding = utf-8

import pandas as pd
import numpy as np
from pandas import Series, DataFrame
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

import f1_user_feature_1 as user



def get_info(df):
    uNum = len(df.uid.unique())
    print "------------------------\nbrowse_history_train表信息\n"
    # print "describe：\n", df_train.describe()
    print "\ninfo:\n", df.info(null_counts=True)
    print "\ncount:\n", df.count()
    print "\nshape:" , df.shape
    print "该表存在的用户数：", uNum
    print "一个用户的记录数是否唯一：", df.uid.is_unique
    print "没有该表记录的用户数：", (len(user.get_train_userid())-uNum)
    print "时间戳缺省的个数为:%s" % len(df[df.time == 0])
    print "用户浏览行为最大值：", max(df.conscnt)
    print "用户浏览行为最大最小差值：", max(df.behaviour) - min(df.behaviour)
    print "表中已有的浏览行为类别数", len(Series(df.behaviour.unique()))
    print "浏览行为最大值与存在的浏览行为类别数差值（表中可能未有的浏览行为类别数）", max(df.behaviour) -len(Series(df.behaviour.unique()))
    print "用户浏览子行为最大值：", max(df.subId)
    print "用户浏览子行为最大最小差值：", max(df.subId) - min(df.subId)






def get_fea():


    df_train = pd.read_table("../data/risk_predict/train/bill_detail_train.txt", sep=',',
                             names=['uid', 'time', 'bid', 'last_money', 'a', 'b', 'c', 'd', 'cons_nt', 'e', 'f', 'e', 'f', 'g', 'h'])
    df_test = pd.read_table("../data/risk_predict/test/bill_detail_test.txt", sep=',',
                             names=['uid', 'time', 'bid', 'last_money', 'a', 'b', 'c', 'd', 'cons_nt', 'e', 'f', 'e', 'f', 'g', 'h'])

    n_uid = user.get_train_userid()
    t_uid = user.get_test_userid()

    return