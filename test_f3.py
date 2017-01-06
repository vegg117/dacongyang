# encoding = utf-8
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import f3_browse_feature_1 as bro


b = np.array(['uid', 'time', 'behaviour', 'subId'])
a = np.array([
    [1, 111, 1, 11],
    [2, 111, 2, 0],
    [3, 111, 1, 0],
    [1, 111, 1, 0],
    [9, 111, 3, 0],
    [2, 111, 8, 20]
])
df_train = DataFrame(a, columns=b)

a = np.array([4,5,6, 7,8,9, 1,2,3])
uid = Series(a)
# print df_train
# print uid



#
# bro.get_info(df_train)
# bro.deal_behavi(df_train)
bro.get_behavi_fea(df_train, df_train, uid, uid)
