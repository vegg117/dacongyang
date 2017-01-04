# coding=utf-8

import time
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
from sklearn import linear_model
from sklearn import cross_validation
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.learning_curve import learning_curve
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import explained_variance_score
from sklearn.metrics import classification_report
from sklearn import preprocessing
from xgboost import XGBClassifier
from xgboost import XGBRegressor

start = time.clock()

df_train_user = pd.read_table("/data/risk_predict/train/user_info_train.txt", sep=',',
                           names=['uid','sex','job','edu','marry','res'])

print df_train_user.columns

df_train_bank = pd.read_table("/data/risk_predict/train/bank_detail_train.txt", sep=',',
                              names=['uid', 'time', 'type', 'money', 'isPayout'])
print df_train_bank.columns


df_test_user = pd.read_table("/data/risk_predict/test/user_info_test.txt", sep=',',
                           names=['uid','sex','job','edu','marry','res'])

df_test_bank = pd.read_table("/data/risk_predict/test/bank_detail_test.txt", sep=',',
                           names=['uid', 'time', 'type', 'money', 'isPayout'])

data_train_overdue = pd.read_table("/data/risk_predict/train/overdue_train.txt", sep=',',
                                   names=['uid', 'y'])
print data_train_overdue.columns

# 数据检查
print "\n检查是否存在空值"
print df_train_bank.info(null_counts=True)
# 统计用户表用户性别缺省值个数
print "用户性别缺省的个数为:%s" % len(df_train_user[df_train_user.sex == 0]) # 用户性别缺省数为1669
# 统计时间为空值的个数（时间戳为0表示空值）
print "时间戳缺省的个数为:%s" % len(df_train_bank[df_train_bank.time == 0])
    # 时间戳缺省个数为38773    数量大，考虑使用其作为标签，使用其他特征估算



# 特征工程

# 用户表特征处理
def deal_user_feature(df):
    dummies_sex = pd.get_dummies(df['sex'], prefix='sex')
    dummies_job = pd.get_dummies(df['job'], prefix='job')
    dummies_edu = pd.get_dummies(df['edu'], prefix='edu')
    dummies_marry = pd.get_dummies(df['marry'], prefix='marry')
    dummies_res = pd.get_dummies(df['res'], prefix='res')
    df = pd.concat([df, dummies_sex, dummies_job, dummies_edu, dummies_marry, dummies_res], axis=1)
    df.drop(['sex', 'job', 'edu', 'marry', 'res'], axis=1, inplace=True)
    return df

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


# 对用户表属性进行处理
df_train_user = deal_user_feature(df_train_user)
# 对银行流水表进行处理
#df_train_bank_origin = df_train_bank
df_train_bank = deal_bank_feature(df_train_bank)
#print "df_train_bank_origin.columns \n%s" % df_train_bank_origin.columns
print "df_train_bank.columns \n%s" % df_train_bank.columns
#print "df_train_bank_origin.head(2) \n%s" % df_train_bank_origin.head(2)
print "df_train_bank.head(2) \n%s" % df_train_bank.head(2)

# 拼接 以uid为键拼接三个表
print "\ndf_train_user的特征个数为：%s" % len(df_train_user.columns)
print "\ndf_train_bank的特征个数为：%s" % len(df_train_bank.columns)
df_train_data = pd.merge(df_train_user, df_train_bank)
print df_train_data.columns
print "\ndf_train_data的特征个数为：%s" % len(df_train_data.columns)
df_train_data = pd.merge(df_train_data, data_train_overdue)
print df_train_data.columns
print "\ndf_train_data的特征个数为：%s" % len(df_train_data.columns)
print df_train_data.head()

# 去掉确定无关属性uid
df_train_data.drop(['uid'], axis=1, inplace=True)

#对测试集做所有相同的特征处理
df_test_user = deal_user_feature(df_test_user)
df_test_bank = deal_bank_feature(df_test_bank)


# 拼接 以uid为键拼接两个表
df_test_data = pd.merge(df_test_user, df_test_bank)
print df_train_data.head()

# 去掉确定无关属性uid，并在此之前将其保存到y_id中
y_id = df_test_data['uid']
print "\ny_id:\n%s" % y_id.head()
df_test_data.drop(['uid'], axis=1, inplace=True)

# 分割训练集的输入和输出(x,y)
y_train = df_train_data['y']
df_train_data.drop(['y'], axis=1, inplace=True)

print "----------------"
print "df_train_data的列属性有：%s\n" % df_train_data.columns
print "df_test_data的列属性有：%s\n" % df_test_data.columns
print "df_train_data.shape:"
print df_train_data.shape
print "df_test_data.shape:"
print df_test_data.shape
print "y_train.shape:"
print y_train.shape
print "y_id.shape:"
print y_id.shape
print "df_train_data type: "
print type (df_train_data)
print "df_test_data type:"
print type (df_test_data)
print "y_train:"
print type (y_train)
print "y_id type:"
print type (y_id)

def get_data():
    """
    :return:
     df_train_data
    """
    return df_train_data, y_train, df_test_data, y_id