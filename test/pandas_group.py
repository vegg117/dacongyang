# coding=utf-8



from pandas import Series, DataFrame
import pandas as pd
import numpy as np
from numpy import nan

a = DataFrame({'A':[1,2,1],
                'B':['a', 'b', 'c'],
               'C':[4, nan, nan],
               'D':[9, 9, 3]
               })

b = DataFrame({'A':[1,2,3],
                'B':['a', 'b', 'c'],
               'C':[4, nan, nan],
               'D':[9, 3, 3]
               })

c = DataFrame({'A':[2,2,3],
                'B':['a', 'b', 'c'],
               'C':[4, nan, nan],
               'D':[9, 3, 3]
               })

print a.columns
d= a.groupby('A').count()
print d

