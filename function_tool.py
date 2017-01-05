# coding=utf-8


def get_modest_data(df):
    '''
    用户本地测试时，缩小数据规模
    :param df:
    :return:
    '''
    maxlen = 2000
    slen = df.shape[0]
    if(slen > maxlen):
        print "数据做了温和的处理.由", slen, "降至", maxlen
        return df[:maxlen]
    else:
        return df
