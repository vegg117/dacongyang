# coding=utf-8

import time
import pandas as pd
import numpy as np
from pandas import Series, DataFrame

import f1_user_feature_1 as user
import f2_bank_feature_1 as bank




userid = user.get_all_userid()

print userid.shape
print userid.head()
print type(userid)
bank.get_bank_feature(userid)
