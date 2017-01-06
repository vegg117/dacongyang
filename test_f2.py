# coding=utf-8

import pandas as pd
import numpy as np
from pandas import Series, DataFrame
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import f2_bank_feature_1 as bank


def test_get_transCount_fea():

    a = np.array([
        [1, 0],
        [1, 1],
        [2, 1],
        [3, 1]
    ])
    print a
    df_train = pd.DataFrame(a, columns=['uid', 'type'])
    print df_train
    df_y = pd.DataFrame({'uid':[1, 2, 3, 4, 5],
                         'y': [1, 0, 0, 1, 0]})

    df_test = pd.DataFrame({'uid':[11, 11, 12, 13],
                             'type': [0, 1, 1, 1]
                            })

    n_userid = df_y['uid'].unique()


    t_userid = pd.DataFrame({'uid': [11, 12, 13, 14, 15, 16]})['uid'].unique()

    print n_userid
    n_trans, t_trans = bank.get_transCount_fea(df_train, df_test, n_userid, t_userid, df_y)






# 测试
test_get_transCount_fea()