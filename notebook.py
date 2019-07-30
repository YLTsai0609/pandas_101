# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# * There are alternative solution : 
# * more readable, more effient (vectorlized)
# * L10

import pandas as pd

# 1. How to import pandas and check the version? 
print(pd.__version__)
print(pd.show_versions(as_json=True))

# +
# 2. How to create a series from a list, numpy array and dict?
import numpy as np
mylist = list('abcedfghijklmnopqrstuvwxyz')
myarr = np.arange(26)
mydict = dict(zip(mylist, myarr))

ser1 = pd.Series(mylist)
ser2 = pd.Series(myarr)
ser3 = pd.Series(mydict)

# +
# 3. How to convert the index of a series into a column of a dataframe?
# L1
mylist = list('abcedfghijklmnopqrstuvwxyz')
myarr = np.arange(26)
mydict = dict(zip(mylist, myarr))
ser = pd.Series(mydict)

df = ser.to_frame().reset_index()
df.head()

# +
# 4. How to combine many series to form a dataframe?
# L1
import numpy as np
ser1 = pd.Series(list('abcedfghijklmnopqrstuvwxyz'))
ser2 = pd.Series(np.arange(26))

df = pd.concat([ser1, ser2], axis=1)
df.head()

# +
# 5. How to assign name to the series’ index?
# L1
ser = pd.Series(list('abcedfghijklmnopqrstuvwxyz'))

ser.name = 'alphabets'

# +
# 6. How to get the items of series A not present in series B?
# L2
ser1 = pd.Series([1, 2, 3, 4, 5])
ser2 = pd.Series([4, 5, 6, 7, 8])

mask = ~ ser1.isin(ser2)
ser1[mask]

# +
# 7. How to get the items not common to both series A and series B?
# L2
ser1 = pd.Series([1, 2, 3, 4, 5])
ser2 = pd.Series([4, 5, 6, 7, 8])

union_ser = pd.Series(np.union1d(ser1, ser2))
intersection_ser = pd.Series(np.intersect1d(ser1, ser2))
xor_ser = union_ser[~ union_ser.isin(intersection_ser)]
xor_ser
# -

# 8. How to get the minimum, 25th percentile, median, 75th, and max of a numeric series?
# L2
ser = pd.Series(np.random.normal(10, 5, 25))
ser.quantile([.0, .25, .5, .75, 1])

# 9. How to get frequency counts of unique items of a series?
#  L1
ser = pd.Series(np.take(list('abcdefgh'), np.random.randint(8, size=30)))
ser.value_counts()

# +
# 10. How to keep only top 2 most frequent values as it is and replace everything else as ‘Other’?
# L2
np.random.RandomState(100)
ser = pd.Series(np.random.randint(1, 5, [12]))

# ALTERNATIVE ANSWER
top_2_frequent = ser.value_counts().nlargest(2).index
ser.where(ser.isin(top_2_frequent), other='Other')
# -

#  11. How to bin a numeric series to 10 groups of equal size?
#  L2
ser = pd.Series(np.random.random(20))
# Note ourput dtype is category
pd.qcut(ser,
        q = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1],
        labels=['1st', '2nd','3rd','4th','5th',
                 '6th', '7th', '8th','9th', '10th'])

# +
# 12. How to convert a numpy array to a dataframe of given shape? (L1)
# L1
ser = pd.Series(np.random.randint(1, 10, 35))

# note : argument -1 will caculate the rest 
# in this case (7, -1) -->  (7 , 35 / 7)
pd.DataFrame(ser.values.reshape(7,-1))

# +
# 13
# -
































