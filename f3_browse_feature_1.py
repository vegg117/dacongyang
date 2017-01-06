# coding=utf-8


import pandas as pd
import numpy as np
from pandas import Series, DataFrame
from sklearn import preprocessing

import f1_user_feature_1 as user


en_behaviCnt = True #




# 数据清洗
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
    print "用户浏览行为最大值：", max(df.behaviour)
    print "用户浏览行为最大最小差值：", max(df.behaviour) - min(df.behaviour)
    print "表中已有的浏览行为类别数", len(Series(df.behaviour.unique()))
    print "浏览行为最大值与存在的浏览行为类别数差值（表中可能未有的浏览行为类别数）", max(df.behaviour) -len(Series(df.behaviour.unique()))
    print "用户浏览子行为最大值：", max(df.subId)
    print "用户浏览子行为最大最小差值：", max(df.subId) - min(df.subId)

    '''
        观察发现：
        train：
            总用户数：55596, 该表用户数：47330，有8266个用户没有记录
            一个用户有多条记录，也不是所有的用户均有记录
            时间没有缺省
            用户浏览行为最大最小差215, 最大值为216
            浏览子行为编号差10，最大值为11
            并不是所有的浏览行为都被操作
    '''
#get_info()



def deal_behavi(df):




    return
    beType = df.behaviour
    print "be_type is unique:", beType.is_unique
    print len(beType)
    u_beType = Series(beType.unique())
    print "be_type is unique:", u_beType.is_unique

    # 统计每种浏览类型的操作次数
    beTypeCnt = Series([])
    for t in u_beType:
        beTypeCnt[t] = len(beType[beType == t])

    # print beTypeCnt.sort_values()

    print "浏览类型数", len(beTypeCnt)
    print "最小类型次数", min(beTypeCnt)
    print "浏览最多的类型与最少的类型的次数的差值", max(beTypeCnt) - min(beTypeCnt)

    # 删除浏览行为发生数少于
    beType = beType

    return

def fun(df, uid, behavi):
    print "------------------"
    # 一个用户可能有多条记录，下面统计每个用户对于每种浏览行为的发生次数
    leng = len(uid)
    # print leng
    for u in uid:
        # print u
        record = df[df['uid'] == u]
        if(record.shape[0] <= 0):
            # 默认为nan
            continue
        print "record:\n", record
        # print type(record) #DF
        leng = record.shape[0]
        brow = record['behaviour']
        k = 0
        print "brow:", brow
        print "u=", u
        while(k < leng):
            print "brow[k]=", brow[k]
            print  behavi[behavi['uid'] == u][record[k]['behaviour']]
            exit()
            behavi[behavi['uid'] == u][brow] = behavi[behavi['uid'] == u][brow]+1
    print behavi
    exit()

    k = 0
    while(k < leng):
        print uid[k]
        print "aaa"
        record = df[df['uid'] == uid[k]]
        print record
    exit()

    return
def get_behavi_fea(df_train, df_test, n_uid, t_uid):

    # 0 获得浏览行为序列，作为特征
    nbh = Series(df_train.behaviour.unique())
    n_behavi = DataFrame(columns=nbh)
    n_behavi['uid'] = n_uid
    n_behavi.drop([1], axis=1, inplace=True)

    tbh = Series(df_test.behaviour.unique())
    t_behavi = DataFrame(columns=tbh)
    t_behavi.drop([1], axis=1, inplace=True)
    t_behavi['uid'] = t_uid

    # print n_behavi.head()
    # print t_behavi.head()
    # exit()
    # 如果 train与test的特征不一致如何处理？

    # 1 获得完整用户的特征信息，无记录的为空值
    fun(df_train, n_uid, n_behavi)

    # 2 用平均值填充浏览行为

    # 3 归一化


    # 特殊处理
    for b in nbh:
        nbh[b] = "behavi_"+str(b)

    tbh = Series(df_test.behaviour.unique())
    for b in tbh:
        tbh[b] = "behavi_" + str(b)
    return


def get_fea():

    # 训练集特征
    n_fea = pd.DataFrame()
    # 测试集特征
    t_fea = pd.DataFrame()

    df_train = pd.read_table("../data/risk_predict/train/browse_history_train.txt", sep=',',
                             names=['uid', 'time', 'behaviour', 'subId'])
    df_test = pd.read_table("../data/risk_predict/test/browse_history_test.txt", sep=',',
                             names=['uid', 'time', 'behaviour', 'subId'])

    n_uid = user.get_train_userid()
    t_uid = user.get_test_userid()
    print type(n_uid)
    exit()
    get_behavi_fea(df_train, df_test, n_uid, t_uid)
    # deal_behavi(df_train)

    return
# get_fea()