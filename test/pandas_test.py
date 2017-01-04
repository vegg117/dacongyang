# coding=utf-8

from pandas import Series, DataFrame
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
