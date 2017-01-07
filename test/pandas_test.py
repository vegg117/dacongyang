# coding=utf-8

from pandas import Series, DataFrame
import pandas as pd
import numpy as np
from numpy import nan
import function_tool as fun


a = np.array([
    [1, 'a', 4, 2],
    [3, 'b', nan, 'c'],
    [1, 'c', nan, 3]
    ])
a = DataFrame(a, columns=['A', 'B', 'C', 'D'])

b = DataFrame({'A':[1,2, nan, 4],
                'B':['a', 'b', nan, 'd'],
               'C':[nan, nan, nan, nan],
               'D':[9, 3, nan, 6]
               })

s1 = pd.Series([5,2,1,4,3])
s2 = pd.Series(['a','v','s','f','e'])
s3 = pd.Series([3,4,5,1,2,6])

c = DataFrame(s1, s2)

arr = np.array([8, -1, 8, -4])



import datetime
import time
starttime = datetime.datetime.now()

time.sleep(10)
k = 0

endtime = datetime.datetime.now()
print (endtime - starttime).total_seconds()

exit()
print a['A'][1]
exit()
for s in s1:
    s3[s] = str(s)+"aa"
print s3

exit()
a = 3
a = str(a) + "ffsadfff"
print a
exit()

exit()
print a.A.is_unique
exit()
print a
print b.info(null_counts=True)
print c.info(null_counts=True)
exit()
print max(a.D)
print max(a)

exit()
arr = arr+1
print arr
exit()

a = 3.4
b  = (a+3)/2
print b
exit()
arr = [8, 8, 8, 8]
b.C = arr
print b

exit()

# 填充
print type(b)
b.fillna(0, inplace = True)
print b
print type(b)
exit()
d1 = pd.DataFrame({'a':s1, 'b':s2, 'c':s3})
print d1
print pd.DataFrame({'a':s1,'c':s3})
exit()
# 缺失值统计
print len(b[b.C.notnull()])
print len(b[b.C.isnull()])
exit()
print b.describe
exit()
print b.count()
exit()
# Series 组合 DF
d = DataFrame({'id':s1, 'name':s2})
print type(d)
print d
exit()
print fun.get_modest_data(b)
exit()
s = pd.Series([])
s[0] = 3
print s[0]
s[3]=4
print s

s1[5] = 999
print s1[4]
print s1[5]
exit()
print len(s1)

print s1[2]
arr1 = s1.as_matrix()

print type(arr1)
print arr1[2]
exit()
arr = np.array([
    [1,10,100],
    [2,12,102],
    [3,13,103],
    [2,14,104]
    ])

d = DataFrame(arr, columns=['y1', 'y2', 'y3'], index=['x1', 'x2', 'x3', 'y4'])
s = Series(2)
print d
print d.shape
print "找出所有y1=2的行  id=2"
print d[d.y1 == 2]
print type (d[d.y1 == 2])
print type (d[d.y1 == 1])
print d[d.y1 == 2].shape
print d[d.y1 == 1].shape
print d[d.y1 == 2]['y2']

exit()
#
print b.C.notnull
t = b.C.notnull
print t
print type(b.C.notnull)
t = b[b.C.notnull()] #返回b的某些行元素(axis=0),这些行要求b.C.notnull对应值为True

print  t
exit()

# 对于一个用户记录表，根据用户的id把记录表分组

# group
res = a[(a.A>=2)]
print res
print type(res)

res = a[(a.A>=2)]['B']
print type(res)
print res
exit()


print type(s1)

exit()

c = [a, b]
print c
print type(c)
print type(c[0])
print c[1]

exit()

print a.shape
print (a.values == b.values)


print "(a[a.D == b.D])"
print (a[a.D == b.D])
print
print a


print "\na.count()%s\n" % a.count()

print "\na.describe()%s\n" % a.describe()
exit()
print a.info(null_counts=True)

print "D列=9个数为:%s" % len(a[a.D == 9])
print (a.D == 9)  #type series

print a[a.D == 9]

print "\nnull value deal"
a['C'].fillna(0, inplace=True)
print a


print a.T


print
