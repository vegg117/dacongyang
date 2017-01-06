# coding=utf-8


import pandas as pd
import numpy as np
from pandas import Series, DataFrame
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

import f1_user_feature_1 as user


en_behaviCnt = True #

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


    beType = df.behaviour
    print "be_type is unique:", beType.is_unique
    print len(beType)
    u_beType = Series(beType.unique())
    print "be_type is unique:", u_beType.is_unique

    # 统计每种浏览类型的发生次数
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

def behavi_fea(df, uid, behavi):
    print "------------------"
    # 一个用户可能有多条记录，下面统计每个用户对于每种浏览行为的发生次数
    # print leng
    for u in uid:
        # print u
        record = df[df['uid'] == u]
        if(record.shape[0] <= 0):
            # 默认为nan
            continue
        # print "record:\n", record
        # print type(record) #DF
        brow = record['behaviour']
        # print "brow:", brow
        # print "u=", u
        for be in brow.unique():
            # print 'be:', be
            # print 'colums:', behavi.columns
            if be in behavi.columns:
                # print 'be:', be
                behavi.loc[behavi['uid'] == u, be] = len(brow[(brow == be)])

    # print 'behavi:\n', behavi
    return behavi
def get_behavi_fea(df_train, df_test, n_uid, t_uid):

    # 0 获得浏览行为序列，作为特征
    nbh = Series(df_train.behaviour.unique())
    n_behavi = DataFrame(columns=nbh)
    n_behavi['uid'] = n_uid
    print n_behavi.head()

    t_behavi = DataFrame(columns=nbh)
    t_behavi['uid'] = t_uid
    print t_behavi.head()

    # print n_behavi.head()
    # print t_behavi.head()
    # exit()
    # 如果 train与test的特征不一致如何处理？

    # 1 获得完整用户的特征信息，无记录的为空值
    # 用户各浏览行为数
    n_be_cnt = behavi_fea(df_train, n_uid, n_behavi)
    n_be_cnt.fillna(0, inplace=True)
    print 'n_be_cnt:\n', n_be_cnt
    t_be_cnt = behavi_fea(df_test, t_uid, t_behavi)
    t_be_cnt.fillna(0, inplace=True)
    print 't_be_cnt:\n', t_be_cnt

    # 用户浏览行为总数
    n_bes_cnt = pd.DataFrame(columns=['uid', 'cnt_total'])
    n_bes_cnt['uid'] = n_be_cnt['uid']
    n_bes_cnt['cnt_total'].fillna(0, inplace=True)
    print n_bes_cnt
    for uid in n_be_cnt['uid']:
        bes = n_be_cnt[n_be_cnt['uid'] == uid].values[0]
        n_bes_cnt.loc[n_bes_cnt['uid'] == uid, 'cnt_total'] = sum(bes[bes > 0]) - uid
    print n_bes_cnt

    t_bes_cnt = pd.DataFrame(columns=['uid', 'cnt_total'])
    t_bes_cnt['uid'] = t_be_cnt['uid']
    t_bes_cnt['cnt_total'].fillna(0, inplace=True)
    print t_bes_cnt
    for uid in t_be_cnt['uid']:
        bes = t_be_cnt[t_be_cnt['uid'] == uid].values[0]
        t_bes_cnt.loc[t_bes_cnt['uid'] == uid, 'cnt_total'] = sum(bes[bes > 0]) - uid
    print n_bes_cnt

    # 去除空值过多于alpha的列
    alpha = len(n_uid) * 0.8
    drop_be = []
    for be in nbh:
        cnt = len(n_be_cnt.loc[n_be_cnt[be] == 0, be])
        # print 'be:', be, cnt
        if cnt > alpha:
            drop_be.append(be)
    print 'drop_be:', drop_be

    # 2 用平均值填充浏览行为
    for be in nbh:
        print 'be:', be
        n_mean = n_be_cnt[be].sum() * 1.0 / sum(n_be_cnt[be] > 0)
        print 'n_mean:', n_mean
        n_be_cnt.loc[n_be_cnt[be] == 0, be] = n_mean
        t_be_cnt.loc[t_be_cnt[be] == 0, be] = n_mean
    print 'n_be_cnt:\n', n_be_cnt
    print 't_be_cnt:\n', t_be_cnt

    n_mean = n_bes_cnt['cnt_total'].sum() * 1.0 / sum(n_bes_cnt['cnt_total'] > 0)
    n_bes_cnt.loc[n_bes_cnt['cnt_total'] == 0, 'cnt_total'] = n_mean
    t_bes_cnt.loc[t_bes_cnt['cnt_total'] == 0, 'cnt_total'] = n_mean
    print 'n_bes_cnt:\n', n_bes_cnt
    print 't_bes_cnt:\n', t_bes_cnt

    # 3 归一化
    print 'n_be_cnt:\n', n_be_cnt
    for be in nbh:
        ss = StandardScaler()
        n_be = ss.fit_transform(n_be_cnt[be])
        t_be = ss.transform(t_be_cnt[be])
        n_be_cnt[be] = n_be
        t_be_cnt[be] = t_be
    print 'n_be_cnt:\n', n_be_cnt

    ss = StandardScaler()
    n_be = ss.fit_transform(n_bes_cnt['cnt_total'])
    t_be = ss.transform(t_bes_cnt['cnt_total'])
    n_bes_cnt['cnt_total'] = n_be
    t_bes_cnt['cnt_total'] = t_be
    print 'n_bes_cnt:\n', n_bes_cnt

    # 特殊处理
    n_be_cnt_fea = pd.DataFrame(n_be_cnt['uid'])
    t_be_cnt_fea = pd.DataFrame(t_be_cnt['uid'])
    print 'n_be_cnt_fea,', n_be_cnt_fea
    for b in nbh:
        print 'b1,', b
        if b not in drop_be:
            print 'b2,', b
            n_be_cnt_fea["behavi_"+str(b)] = n_be_cnt[b]
            t_be_cnt_fea["behavi_"+str(b)] = t_be_cnt[b]

    print n_be_cnt_fea
    print t_be_cnt_fea
    return n_be_cnt_fea, t_be_cnt_fea, n_bes_cnt, t_bes_cnt


def get_fea():



    df_train = pd.read_table("../data/risk_predict/train/browse_history_train.txt", sep=',',
                             names=['uid', 'time', 'behaviour', 'subId'])
    df_test = pd.read_table("../data/risk_predict/test/browse_history_test.txt", sep=',',
                             names=['uid', 'time', 'behaviour', 'subId'])

    n_uid = user.get_train_userid()
    t_uid = user.get_test_userid()
    print type(n_uid)

    n_be_cnt_fea, t_be_cnt_fea, n_bes_cnt, t_bes_cnt = get_behavi_fea(df_train, df_test, n_uid, t_uid)

    n_fea = pd.merge(n_be_cnt_fea, n_bes_cnt, on='uid')
    t_fea = pd.merge(t_be_cnt_fea, t_bes_cnt, on='uid')

    print 'n_fea:\n', n_fea
    return n_fea, t_fea
# get_fea()