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

# ## There are alternative solution and hits: 
# ### more readable
# > 10, 18, 23, 24, 25
# ### more effient (vectorlized)
# > 16, 19, 24
# ### when to use it?
# > 13, 26
# ### hints
# > 12, 18, 22, 24

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
# More Readable
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
# 13. How to find the positions of numbers that are multiples of 3 from a series?
# L2
ser = pd.Series(np.random.randint(1, 10, 7))

# note : np.where, pd.where return whole series
#        np.argwhere return index
#        use the indrx :  arr = np.argwhere(ser condition), ser.iloc(arr.reshape(-1))
np.argwhere(ser % 3 == 0)

# +
# 14. How to extract items at given positions from a series
# L1
ser = pd.Series(list('abcdefghijklmnopqrstuvwxyz'))
pos = [0, 4, 8, 14, 20]

ser.iloc[pos]

# +
# 15. How to stack two series vertically and horizontally ?
# Difficulty Level: L1
ser1 = pd.Series(range(5))
ser2 = pd.Series(list('abcde'))

df1 = pd.concat([ser1,ser2], axis=0).to_frame()
df2 = pd.concat([ser1,ser2], axis=1)

# +
# 16. How to get the positions of items of series A in another series B?
# Difficulty Level: L2
ser1 = pd.Series([10, 9, 6, 5, 3, 1, 12, 8, 13])
ser2 = pd.Series([1, 3, 10, 13])

# note : this solution is vectorlized
# faster than list comprehemsion for i in ser2 
# when data is big
np.argwhere(ser1.isin(ser2)).reshape(-1).tolist()
# + {}
# 17. How to compute the mean squared error on a truth and predicted series?
# Difficulty Level: L2
truth = pd.Series(range(10))
pred = pd.Series(range(10)) + np.random.random(10)

print((truth - pred).pow(2).sum() / len(truth))
print(np.mean((truth - pred) ** 2))

# +
# 18. How to convert the first character of each element in a series to uppercase?
# Difficulty Level: L2
ser = pd.Series(['how', 'to', 'kick', 'ass?'])

# More readable
ser.str.capitalize()
# Hints
# you can use dir() to get all the method inside ser.str
# Now you could faster understand how ser.str can do
# print(dir(ser.str))

# +
# 19. How to calculate the number of characters in each word in a series?
# Difficulty Level: L2
ser = pd.Series(['how', 'to', 'kick', 'ass?'])

# vectorlzied

ser.str.len()

# +
# 20. How to compute difference of differences between consequtive numbers of a series?
# Difficulty Level: L1
ser = pd.Series([1, 3, 6, 10, 15, 21, 27, 35])

# Solition 1
tmp = ser.shift(1)
middle_result = ser - tmp
print(middle_result.tolist())
tmp2 = middle_result.shift(1)
print((middle_result - tmp2).tolist())

# Solution 2
print('-'*60)
print(ser.diff().tolist())
print(ser.diff().diff().tolist())
# + {}
# 21. How to convert a series of date-strings to a timeseries?
# Difficiulty Level: L2

ser = pd.Series(['01 Jan 2010', '02-02-2011', '20120303', '2013/04/04', '2014-05-05', '2015-06-06T12:20'])

pd.to_datetime(ser, infer_datetime_format=True)

# +
# 22. How to get the day of month, week number, day of year and day of week from a series of date strings?
# Difficiulty Level: L2

ser = pd.Series(['01 Jan 2010', '02-02-2011', '20120303', '2013/04/04', '2014-05-05', '2015-06-06T12:20'])

tmp = pd.to_datetime(ser)
# hint
# use dir(tmp.dt) to check it out what could be called
# tmp.dt is pandas.core.indexes.accessors.DatetimeProperties object
# use dir(tmp[0]) to check it out what could be called 
# tmp[0] is a single element, <class 'pandas._libs.tslibs.timestamps.Timestamp'>
# you might wanna to check
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.html
# for Timestamp documentation
# https://dateutil.readthedocs.io/en/stable/index.html
# dateutil, really good when we want to dealing with time


Date = tmp.dt.day.tolist()
Week_number = tmp.dt.weekofyear.tolist()
Day_num_of_year = tmp.dt.dayofyear.tolist()
Dayofweek = tmp.dt.weekday_name.tolist()

print(f''' 
Date : {Date}
Week number : {Week_number}
Day num of year : {Day_num_of_year}
Day of week : {Dayofweek}
''')


# +
# 23. How to convert year-month string to dates corresponding to the 4th day of the month?
# Difficiulty Level: L2
ser = pd.Series(['Jan 2010', 'Feb 2011', 'Mar 2012'])

# ALTERNATIVE SOLUTION
# more readable
pd.to_datetime(ser, infer_datetime_format=True)

# +
# 24. How to filter words that contain atleast 2 vowels from a series?
# Difficiulty Level: L3

ser = pd.Series(['Apple', 'Orange', 'Plan', 'Python', 'Money'])

# ALTERNATIVE SOLUTION
# more readable
# vectorlized
condition = ser.str.count('[aeiouAEIOU]') >= 2
ser[condition]
# Hint, you cloud use print(dir(ser.str)) to check it out what could be called

# -

# * hint
#     * <img src = "./RegExp_snap.png"></img>
#     * pandas中的Series.str方法是向量化的，且通常都支援正則表達式(RegExp)
#     * 正則表達式可以幫助我們處理很多文字問題
#     * 像圖中的[aeiou]搭配[AEIUO] --> [aeiuoAEIUO]就解決了此題
#     * 或許你會想看看這份在[菜鳥上的教學](http://www.runoob.com/python/python-reg-expressions.html)


# +
# 25. How to filter valid emails from a series?
# Difficiulty Level: L3

# Extract the valid emails from the series emails. The regex pattern for valid emails is provided as reference.

emails = pd.Series(['buying books at amazom.com', 'rameses@egypt.com', 'matt@t.co', 'narendra@modi.com'])
pattern ='[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,4}'

# More readable
condition = emails.str.contains(pattern)
emails[condition].tolist()

# +
# 26. How to get the mean of a series grouped by another series?
# Difficiulty Level: L2
fruit = pd.Series(np.random.choice(['apple', 'banana', 'carrot'], 10))
weights = pd.Series(np.linspace(1, 10, 10))
print(weights.tolist())
print(fruit.tolist())

print()
print(weights.groupby(fruit).mean())

# ALTERNATIVE SULOTION
tmp = pd.DataFrame({'fruit':fruit, 
              'weights':weights}).groupby('fruit').mean()

tmp['weights'].index.name = ''
print(tmp['weights'])

# When to use?
# 操作dataframe時我們通常都直接在dataframe裡面groupby, 這題告訴我們
# series 可以 groupby 另一條series, 之間用index作為對應, 
# 這讓feature engineering時能夠有更好的彈性
# + {}
# 27. How to compute the euclidean distance between two series?
# Difficiulty Level: L2

p = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
q = pd.Series([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])

sum((p - q) ** 2) ** 0.5


# +
# 28. How to find all the local maxima (or peaks) in a numeric series?
# Difficiulty Level: L3

# Get the positions of peaks (values surrounded by smaller values on both sides) in ser.

ser = pd.Series([2, 10, 3, 4, 9, 10, 2, 7, 3])

# more readable
peak_locs = np.argwhere(np.sign(ser.diff(1)) +\
                        np.sign(ser.diff(-1)) == 2).reshape(-1) 

# hint
# use diff(-1) to get the backforwd diff
# you could use this when using diff, shift function
# diff
peak_locs

# +
# 29. How to replace missing spaces in a string with the least frequent character?
# Replace the spaces in my_str with the least frequent character.

# Difficiulty Level: L2

# hint 
# a way to concant all series strnig 
ser = pd.Series(list('dbc deb abed gade'))
freq = ser.value_counts()
print(freq)
least_freq = freq.dropna().index[-1]
result = "".join(ser.replace(' ', least_freq))
print(result)

# ALTERNATIVE SOLUTION
my_str = 'dbc deb abed gade'

least_freq_character = pd.Series(list(my_str)).value_counts().index[-1]

my_str.replace(' ', least_freq_character)


# +
# 30. How to create a TimeSeries starting ‘2000-01-01’ and 10 weekends (saturdays) after that having random numbers as values?
# Difficiulty Level: L2
dateime_idx = pd.date_range('2000-01-01', periods=10, freq='W-SAT')

pd.Series(index = dateime_idx,
          data = np.random.randint(2,8,size=len(dateime_idx)))
# hint 
# it is really hard to find the pd.date_range method in documentation
# here is the documentation
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.date_range.html


# -


