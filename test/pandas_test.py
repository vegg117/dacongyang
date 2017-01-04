# coding=utf-8

from pandas import Series, DataFrame
import pandas as pd
import numpy as np
from numpy import nan

a = DataFrame({'A':[1,2,3],
                'B':['a', 'b', 'c'],
               'C':[4, nan, nan],
               'D':[9, 9, 3]
               })

b = DataFrame({'A':[1,2,3],
                'B':['a', 'b', 'c'],
               'C':[4, nan, nan],
               'D':[9, 3, 3]
               })

s1 = pd.Series([1,2,3,4,5])
s1 = pd.Series([1,2,3,4,5])

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
