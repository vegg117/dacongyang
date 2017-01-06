# encoding = utf-8
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import f3_browse_feature_1 as bro
from sklearn.preprocessing import StandardScaler


def test_get_behavi_fea():
    b = np.array(['uid', 'time', 'behaviour', 'subId'])
    a = np.array([
        [1, 111, 1, 11],
        [2, 111, 2, 0],
        [9, 111, 1, 0],
        [9, 111, 1, 1],
        [9, 111, 3, 0],
        [2, 111, 8, 20]
    ])
    df_train = DataFrame(a, columns=b)
    print "df_train:\n", df_train.head()

    b = np.array(['uid', 'time', 'behaviour', 'subId'])
    a = np.array([
        [11, 111, 1, 11],
        [11, 111, 2, 0],
        [19, 111, 1, 0],
        [11, 111, 11, 20],
        [11, 111, 2, 0]
    ])
    df_test = DataFrame(a, columns=b)

    a = np.array([4,5,6, 7,8,9, 1,2,3, 10])
    uid = Series(a)
    # print df_train
    print uid

    a = np.array([11, 19, 18])
    t_uid = Series(a)
    print 't_uid:', t_uid

    #
    # bro.get_info(df_train)
    # bro.deal_behavi(df_train)
    n_be_cnt_fea, t_be_cnt_fea, n_bes_cnt, t_bes_cnt = bro.get_behavi_fea(df_train, df_test, uid, t_uid)
    n_fea = pd.merge(n_be_cnt_fea, n_bes_cnt, on='uid')
    print 'n_fea:\n', n_fea

def stdscaler():
    b = np.array(['uid', 'time', 'behaviour', 'subId'])
    a = np.array([
        [1, 111, 1, 11],
        [2, 111, 2, 0],
        [9, 111, 1, 0],
        [9, 111, 1, 1],
        [9, 111, 3, 0],
        [2, 111, 8, 20]
    ])
    df_train = DataFrame(a, columns=b)
    ss = StandardScaler()
    d1 = ss.fit_transform(df_train['behaviour'])
    print d1


test_get_behavi_fea()
# stdscaler()