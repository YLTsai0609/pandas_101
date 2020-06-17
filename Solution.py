# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ## There are alternative solution and hits: 
# ### more readable
# > 10, 18, 23, 24, 25, 28, 37, 38, 44, 54, 58, 60, 66, 71
# ### more effient (vectorlized)
# > 16, 19, 24, 43, 62
# ### when to use it?
# > 13, 26, 36, 39, 41, 58, 60, 67
# ### hints
# > 12, 18, 22, 24, 28, 29, 30, 33, 56, 59, 66, 67, 72

# # Pandas 101

import pandas as pd
from IPython.core.display import display

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

# More Readable
top_2_frequent = ser.value_counts().nlargest(2).index
ser.where(ser.isin(top_2_frequent), other='Other')

# +
#  11. How to bin a numeric series to 10 groups of equal size?
#  L2
ser = pd.Series(np.random.random(20))

pd.qcut(ser,
        q = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1],
        labels=['1st', '2nd','3rd','4th','5th',
                 '6th', '7th', '8th','9th', '10th'])

# hint 
# 使用qcut輸出的series, dtype為categorical, 各個value之間可以進行比較

# +
# 12. How to convert a numpy array to a dataframe of given shape? (L1)
# L1
ser = pd.Series(np.random.randint(1, 10, 35))


pd.DataFrame(ser.values.reshape(7,-1))
# hint
# 參數 -1 --> 處理好剩下的dimension，在此例中，(7,-1) --> (7, 35/7)

# +
# 13. How to find the positions of numbers that are multiples of 3 from a series?
# L2
ser = pd.Series(np.random.randint(1, 10, 7))



# np.argwhere(ser % 3 == 0)

# hint
# np.where, pd.where 傳回整個series
# np.argwhere 傳回index

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

df1 = pd.concat([ser1,ser2], axis='index').to_frame()
df2 = pd.concat([ser1,ser2], axis='columns')
display(df1.head(), df2)

# +
# 16. How to get the positions of items of series A in another series B?
# Difficulty Level: L2
ser1 = pd.Series([10, 9, 6, 5, 3, 1, 12, 8, 13])
ser2 = pd.Series([1, 3, 10, 13])

# vectorlized
mask = ser1.isin(ser2)
np.argwhere(mask.values).reshape(-1).tolist()

# np.where進行項量化的操作, 在資料量大時可以保證一定的速度
# +
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
# 使用dir來得到所有 ser.str中的屬性及方法
# 如此一來可以更全面的了解有什麼屬性及方法可以call
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
# +
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
# 使用 dir(tmp.dt) 來確認有什麼屬性/方法可以call
# tmp.dt為pandas.core.indexes.accessors.DatetimeProperties 物件
# 使用dir(tmp[0])  來確認有什麼屬性/方法可以call
# tmp[0]為<class 'pandas._libs.tslibs.timestamps.Timestamp'> 物件
#  pandas timestamp文件
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.html
#  dateutil 文件，處理時間資料時非常常用
# https://dateutil.readthedocs.io/en/stable/index.html



Date = tmp.dt.day.tolist()
Week_number = tmp.dt.weekofyear.tolist()
Day_num_of_year = tmp.dt.dayofyear.tolist()
Dayofweek = tmp.dt.weekday.tolist()

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

# more readable
pd.to_datetime(ser, infer_datetime_format=True)

# +
# 24. How to filter words that contain atleast 2 vowels from a series?
# Difficiulty Level: L3

ser = pd.Series(['Apple', 'Orange', 'Plan', 'Python', 'Money'])


# more readable
# vectorlized
condition = ser.str.count('[aeiouAEIOU]') >= 2
ser[condition]


# -

# * hint
#     * <img src = "./images/RegExp_snap.png"></img>
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
# +
# 27. How to compute the euclidean distance between two series?
# Difficiulty Level: L2

p = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
q = pd.Series([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])

sum((p - q) ** 2) ** 0.5


# +
# # 28. How to find all the local maxima (or peaks) in a numeric series?
# # Difficiulty Level: L3

# # Get the positions of peaks (values surrounded by smaller values on both sides) in ser.

# ser = pd.Series([2, 10, 3, 4, 9, 10, 2, 7, 3])

# # more readable
# peak_locs = np.argwhere(np.sign(ser.diff(1)) +\
#                         np.sign(ser.diff(-1)) == 2).reshape(-1) 

# # hint
# # 使用diff(-1)，來取得backward
# # 同樣性質的方法還有pd.Series.shift
# peak_locs

# +
# 29. How to replace missing spaces in a string with the least frequent character?
# Replace the spaces in my_str with the least frequent character.

# Difficiulty Level: L2

# hint 
# 此例示範了如何將所有series中的string coacat在一起

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
# 要在pandas中找到pd.date_range這個方法實在很難找......
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.date_range.html


# +
# 30 - 2
# create date index / date column - with dataframe length
from datetime import datetime, timedelta

data = np.random.randint(2, 8, size=100)

date_start = datetime(2018, 1, 1)
days = pd.date_range(date_start,
                     date_start + timedelta(len(data) - 1),
                     freq='D') # sart, end, freq



# days = pd.date_range('2018-01-01', periods=1, freq='D')
df = pd.DataFrame({'value':data,'date':days})

display(df.head(),
       df.dtypes,
       df.set_index('date'))


# +
# 31. How to fill an intermittent time series so all missing dates show up with values of previous non-missing date?
# Difficiulty Level: L2

ser = pd.Series([1,10,3,np.nan], index=pd.to_datetime(['2000-01-01', '2000-01-03', '2000-01-06', '2000-01-08']))

# Fill the datetime index using resample + ffill()
ser.resample('D').ffill()


# +
# 32. How to compute the autocorrelations of a numeric series?
# Difficiulty Level: L3

# Compute autocorrelations for the first 10 lags of ser. Find out which lag has the largest correlation.

ser = pd.Series(np.arange(20) + np.random.normal(1, 10, 20))

# ALTERNATIVE　SOLUTION
tmp = pd.Series([abs(ser.autocorr(lag)) for lag in range(1,11)])
tmp.sort_values(ascending=False).head(1)

# +
# 33. How to import only every nth row from a csv file to create a dataframe?
# Difficiulty Level: L2

# Import every 50th row of BostonHousing dataset as a dataframe.
url = 'https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv'



df = pd.read_csv(url, chunksize=50)

df_result = pd.concat([chunk.iloc[0] for chunk in df], axis=1)

df_result.T
# Hint
# pd.read_csv(**param, chunksize=50)
# 會return TextFileReader物件, 是可遞迴物件
# 可以經由以下確認
# print(type(df))


# +
# 34. How to change column values when importing csv to a dataframe?
# Difficulty Level: L2

# Import the boston housing dataset, but while importing change the 'medv' (median house value) column so that values < 25 becomes ‘Low’ and > 25 becomes ‘High’.

url = 'https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv'
converters = {'medv': lambda x :'High' if float(x) > 25
                                       else 'Low'}
df = pd.read_csv(url, converters=converters)
df.head()



# +
# 35. How to create a dataframe with rows as strides from a given series?
# Difficiulty Level: L3

L = pd.Series(range(15))

def gen_strides(a, stride_len=5, window_len=5):
    n_strides = ((a.size-window_len)//stride_len) + 1
    return np.array([a[s:(s+window_len)] for s in np.arange(0, a.size, stride_len)[:n_strides]])

gen_strides(L, stride_len=2, window_len=4)

# +
# 36. How to import only specified columns from a csv file?
# Difficulty Level: L1

url = 'https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv'
cols = ['crim','medv']
pd.read_csv(url, usecols=cols).head()

# When to use
# 資料量大時，硬體記憶體不足，只讀取幾個column做特徵工程
# 並存取特徵結果

# +
# 37. How to get the nrows, ncolumns, datatype, summary stats of each column of a dataframe? Also get the array and list equivalent.
# Difficulty Level: L2
url = 'https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv'
df = pd.read_csv(url)

# More readable
display(
df.shape,
df.dtypes,
df.describe()
)
# get np.array and list
array = df.values
df_list = df.values.tolist()


# +
# # 38. How to extract the row and column number of a particular cell with given criterion?
# # Difficulty Level: L1

# df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')

# # More Readable
# highest_price = df['Price'].max()
# # the dataframe rows
# df.query(f'Price == {highest_price}')
# # the row idx and col idx
# row, col = np.argwhere(df.values == np.max(df.Price)).reshape(-1)
# print(row, col)
# +
# # 39. How to rename a specific columns in a dataframe?
# # Difficulty Level: L2

# # Rename the column Type as CarType in df and replace the ‘.’ in column names with ‘_’.
# df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')

# tmp = [col.replace('.','_') for col in df.columns]
# df.columns = tmp
# df = df.rename(columns={'Type':'CarType'})
# df.columns

# # When to use
# # 第一次拿到資料時，針對特徵欄位做資料清理
# # 甚至會加註categorical feature CAT_FeatureName
# # Numerical feature Num_FeatureName......等


# +
# # 40. How to check if a dataframe has any missing values?
# # Difficulty Level: L1
# df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')

# df.isnull().any().any()

# +
# # 41. How to count the number of missing values in each column?
# # Difficulty Level: L2
# df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')

# nan_p_series = df.isnull().sum() / len(df)
# nan_p_series.sort_values(ascending=False).head(1)

# # when to use
# # 缺失值統計，近乎每次必用

# +
# # 42. How to replace missing values of multiple numeric columns with the mean?
# # Difficulty Level: L2

# df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')

# for col in ['Min.Price','Max.Price']:
#     respective_mean = df[col].mean()
#     df[col] = df[col].fillna(respective_mean)

# df[['Min.Price','Max.Price']].isnull().any().any()

# +
# # 43. How to use apply function on existing columns with global variables as additional arguments?
# # Difficulty Level: L3

# # In df, use apply method to replace the missing values in Min.Price with the column’s mean and those in Max.Price with the column’s median.

# df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')

# d = {'Min.Price': np.nanmean, 'Max.Price': np.nanmedian}
# df[['Min.Price', 'Max.Price']] = df[['Min.Price', 'Max.Price']]\
# .apply(lambda x, d: x.fillna(d[x.name](x)), args=(d, ))

# # Vectorlized 
# # 使用 42題的方式，改成median填入，是向量化操作
# # 在資料量大時會快非常多

# +
# 44. How to select a specific column from a dataframe as a dataframe instead of a series?
# Difficulty Level: L2

df = pd.DataFrame(np.arange(20).reshape(-1, 5), columns=list('abcde'))

# More readable

df[['a']]

# +
# 45. How to change the order of columns of a dataframe?
# Difficulty Level: L3
# Actually 3 questions.

# In df, interchange columns 'a' and 'c'.
# Create a generic function to interchange two columns, without hardcoding column names.

# Sort the columns in reverse alphabetical order, that is colume 'e' first through column 'a' last.

df = pd.DataFrame(np.arange(20).reshape(-1, 5), columns=list('abcde'))

# 1
df[list('cbade')]

# 2 
# no function to do that
def swap_col(df, col1, col2):
    df_swap = df.copy()
    col_list = list(df_swap.columns)
    col1_idx = col_list.index(col1)
    col2_idx = col_list.index(col2)
    col_list[col1_idx] = col2
    col_list[col2_idx] = col1
    
    return df_swap[col_list]

swap_col(df, 'c','e')

# 3
new_order = sorted(list(df.columns), reverse=True)
df[new_order]

# +
# 46. How to set the number of rows and columns displayed in the output?
# Difficulty Level: L2

# Change the pamdas display settings on printing the dataframe df it shows a maximum of 10 rows and 10 columns.

df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')
# 找到所有可用的set_option
# pd.describe_option()
pd.set_option('display.max_columns', 10) 
pd.set_option('display.max_rows', 10) 
df
# -
pd.describe_option()

# +
# 47. How to format or suppress scientific notations in a pandas dataframe?
# Difficulty Level: L2

# Suppress scientific notations like ‘e-03’ in df and print upto 4 numbers after decimal.
df = pd.DataFrame(np.random.random(4)**10, columns=['random'])


pd.options.display.float_format = '{:,.4f}'.format

display(df)

# undo
pd.options.display.float_format = None


# +
# 48. How to format all the values in a dataframe as percentages?
# Difficulty Level: L2

# Format the values in column 'random' of df as percentages.

df = pd.DataFrame(np.random.random(4), columns=['random'])


df.style.format({'random':'{0:.2%}'.format,})

# +
# 49. How to filter every nth row in a dataframe?
# Difficulty Level: L1

# From df, filter the 'Manufacturer', 'Model' and 'Type' for every 20th row starting from 1st (row 0).

df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')

# More readable

idx = np.arange(0, df.shape[0], step=20)
col = ['Manufacturer','Model','Type']
df[col].iloc[idx]

# +
# # 50. How to create a primary key index by combining relevant columns?
# # Difficulty Level: L2

# # In df, Replace NaNs with ‘missing’ in columns 'Manufacturer', 'Model' and 'Type' 
# # and create a index as a combination of these three columns and check if the index is a primary key.

# df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv', usecols=[0,1,2,3,5])

# col = ['Manufacturer','Model','Type']
# idx_col = '.Price'
# df[col] = df[col].fillna('missing')
# df.index = df.Manufacturer + '_' + df.Model + '_' + df.Type
# display(df.head(),
#        df.index.is_unique)
# # When to use
# # 根據column內容來設置 Primary key的手法非常實用
# # 容易給其他同事進行表格閱讀

# +
# 51. How to get the row number of the nth largest value in a column?
# Difficulty Level: L2
# Find the row position of the 5th largest value of column 'a' in df.
df = pd.DataFrame(np.random.randint(1, 30, 30).reshape(10,-1), columns=list('abc'))

# More readable

df['a'].nlargest().index[-1]

# +
# 52. How to find the position of the nth largest value greater than a given value?
# Difficulty Level: L2

# In ser, find the position of the 2nd largest value greater than the mean.

# More readable
ser = pd.Series(np.random.randint(1, 100, 15))

tmp = ser - ser.mean()

tmp.nlargest(2).index

# +
# 53. How to get the last n rows of a dataframe with row RowSum > 100?
# Difficulty Level: L2

# Get the last two rows of df whose row RowSum is greater than 100.
df = pd.DataFrame(np.random.randint(10, 40, 60).reshape(-1, 4))

# More readable
df['RowSum'] = df.sum(axis=1)
df.query('RowSum > 100').tail(2)
# +
# 54. How to find and cap outliers from a series or dataframe column?
# Difficulty Level: L2

# Replace all values of ser in the lower 5%ile and greater than 95%ile with respective 5th and 95th %ile value.

# More readable
ser = pd.Series(np.logspace(-2, 2, 30))
low = np.quantile(ser, q=.05)
high = np.quantile(ser, q=.95)
ser.clip(low, high)


# +
# 55. How to reshape a dataframe to the largest possible square after removing the negative values?

# Difficulty Level: L3

# Reshape df to the largest possible square with negative values removed.
# Drop the smallest values if need be. 
# The order of the positive numbers in the result should remain the same as the original.

df = pd.DataFrame(np.random.randint(-20, 50, 100).reshape(10,-1))
display(df)
def larget_square(df):
    tmp = df.copy()
    # get positive array
    arr = tmp[tmp > 0].values
    arr = arr[~ np.isnan(arr)]
    # get largest square of side
    N = np.floor(arr.shape[0] ** 0.5).astype(int)
    # get index the arr soted
    top_idx = np.argsort(arr)[::-1]
    # drop min idx satisfied largest square
    # then reshape
    # key point, sort the top_idx, will give us 
    # the original idx already take out minmimum value
    filtered_idx = top_idx[: (N ** 2)]
    result = pd.DataFrame(arr[sorted(filtered_idx)].reshape(N, -1))
    return result

larget_square(df)

# +
# 56. How to swap two rows of a dataframe?
# Difficulty Level: L2

# Swap rows 1 and 2 in df.

df = pd.DataFrame(np.arange(25).reshape(5, -1))
display(df.head(2))
row1, row2 = df.iloc[0].copy(), df.iloc[1].copy()
df.iloc[0], df.iloc[1] = row2, row1
display(df.head(2))

# Hint
# 使用copy, 否則你的row1, row2和原本的dataframe是連動的

# +
# 57. How to reverse the rows of a dataframe?
# Difficulty Level: L2

# Reverse all the rows of dataframe df.

df = pd.DataFrame(np.arange(25).reshape(5, -1))

df.iloc[::-1]

# +
# 58. How to create one-hot encodings of a categorical variable (dummy variables)?
# Difficulty Level: L2

# Get one-hot encodings for column 'a' in the dataframe df and append it as columns.

df = pd.DataFrame(np.arange(25).reshape(5,-1), columns=list('abcde'))

# More readable
concat_list = [df.drop(columns=['a']),
              pd.get_dummies(df['a'])]

result = pd.concat(concat_list, axis=1)
result

# When to use
# one hot encoding 非常常用, 其中get_dummy有sparse, drop_first可以選
# 但testing set 或是 validation set 出現unseen column 需要進行處理
# 推薦使用 sklearn.preprocessing, unseen col會使training col 全為0
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html


# +
# 59. Which column contains the highest number of row-wise maximum values?
# Difficulty Level: L2

# Obtain the column name with the highest number of row-wise maximum’s in df.


df = pd.DataFrame(np.random.randint(1,100, 40).reshape(10, -1))
display(df)
def largest_value_col_row_wise(df):
    tmp = df.copy()
    result_col = []
    for row in range(tmp.shape[0]):
        max_idx = tmp.iloc[row,:].idxmax()
        result_col.append(tmp.columns[max_idx])
    return result_col
result = largest_value_col_row_wise(df)
result

# Hint, pd.Series.argmax 要被棄用了 使用idxmax替代


# +
# 60. How to create a new column that contains the row number of nearest column by euclidean distance?
# Create a new column such that, each row contains the row number of nearest row-record by euclidean distance.

# Difficulty Level: L3

df = pd.DataFrame(np.random.randint(1,100, 40).reshape(10, -1), columns=list('pqrs'), index=list('abcdefghij'))
display(df.head())

# more readable
def get_neast_and_euclidean_dist(df):
    from scipy.spatial.distance import pdist, squareform
    row_name = df.index.tolist()
    dist = pdist(df, 'euclidean')
    df_dist = pd.DataFrame(squareform(dist), columns=row_name, index=row_name)
    
    nearest_list = []
    nearest_dist_list = []
    # instead of list comprehension
    # for loop is more readable for complex operation
    for row in df_dist.index:
        nearest_info = df_dist.loc[row, :].sort_values()
        nearest_idx, nearest_dist = nearest_info.index[1], nearest_info.values[1]
        nearest_list.append(nearest_idx)
        nearest_dist_list.append(nearest_dist)
    df_dist['nearset'] = nearest_list
    df_dist['dist'] = nearest_dist_list
    
    return df_dist
get_neast_and_euclidean_dist(df)

# when to use
# 計算距離時, pdist, squareform
# 提供了非常多基於numpy計算的距離，包含euclidean, cosine, correlation,
# hamming, 等等, 非常實用
# +
# 61. How to know the maximum possible correlation value of each column against other columns?
# Difficulty Level: L2

# Compute maximum possible absolute correlation value of each column against other columns in df.

df = pd.DataFrame(np.random.randint(1,100, 80).reshape(8, -1), columns=list('pqrstuvwxy'), index=list('abcdefgh'))

corr_df = abs(df.corr())

max_corr_list = [(feature, 
                  (corr_df[feature].sort_values(ascending=False).index[1],
                   corr_df[feature].sort_values(ascending=False)[1]))
                 for feature in corr_df.columns]
max_corr_list


# +
# 62. How to create a column containing the minimum by maximum of each row?
# Difficulty Level: L2

# Compute the minimum-by-maximum for every row of df.

df = pd.DataFrame(np.random.randint(1,100, 80).reshape(8, -1))

# Vectorlized

df['RowMinByMax'] = df.min(axis=0) / df.max(axis=0)
df.head()

# +
# 63. How to create a column that contains the penultimate value in each row?
# Difficulty Level: L2

# Create a new column 'penultimate' which has the second largest value of each row of df.
df = pd.DataFrame(np.random.randint(1,100, 80).reshape(8, -1))

row_penultimate_collection = []
for row in df.index:
    row_penultimate = df.loc[row,:].sort_values().iloc[1]
    row_penultimate_collection.append(row_penultimate)
df['Row_Penultimate'] = row_penultimate_collection
df.head()

# +
# 64. How to normalize all columns in a dataframe?
# Difficulty Level: L2
df = pd.DataFrame(np.random.randint(1,100, 80).reshape(8, -1))

# vectorlized
def Nomolize_df(df, minmax=False):
    tmp = df.copy()
    for col in tmp.columns:
        if not minmax:
            mean = tmp[col].mean()
            std = tmp[col].std()
            tmp[col] = (tmp[col] - mean) / std
        min_col = tmp[col].min()
        max_col = tmp[col].max()
        tmp[col] = (max_col - tmp[col]) / (max_col - min_col)
    return tmp

df_nor = Nomolize_df(df)
df_minmax = Nomolize_df(df, minmax=True)
display(df_nor,
       df_minmax)

# +
# 65. How to compute the correlation of each row with the suceeding row?
# Difficulty Level: L2

# Compute the correlation of each row of df with its succeeding row.

df = pd.DataFrame(np.random.randint(1,100, 80).reshape(8, -1))

corr_list = []
for first_row_idx, second_row_idx in zip(df.index[:-1],
                                         df.index[1:]):
    first_row = df.iloc[first_row_idx,:]
    second_row = df.iloc[second_row_idx,:]

    corr_list.append(
        (first_row_idx, second_row_idx, first_row.corr(second_row))
    )
corr_list

# +
# 66. How to replace both the diagonals of dataframe with 0?
# Difficulty Level: L2

# Replace both values in both diagonals of df with 0.

df = pd.DataFrame(np.random.randint(1,100, 100).reshape(10, -1))

# More readable
np.fill_diagonal(df.values, 0)

df
# hint
# np.fill_diagonal 沒有return值, 
# 因此 new_array = np.fill_diagonal(df.values, 0), new_array會為None
# pd.DataFrame.values 只能呼叫，無法直接帶入值
# 因此 df.vales = np.fill_diagonal(df.values, 0) 不會work
# +
# 67. How to get the particular group of a groupby dataframe by key?
# Difficulty Level: L2

df = pd.DataFrame({'col1': ['apple', 'banana', 'orange'] * 3,
                   'col2': np.random.rand(9),
                   'col3': np.random.randint(0, 15, 9)})

df_grouped = df.groupby(['col1'])


df_grouped.get_group('apple')

# hint 使用 # print(dir(df_grouped)) 來呼叫groupby物件的所有屬性及方法
# hint 使用 以下 來呼叫groupby中所有非隱藏方法
# print([method for method in dir(df_grouped)
#                          if not method.startswith('_')])
# when to use
# 對於groupby的操作，在資料處理時使用頻率非常高，有了上面的hint
# 我們可以更有依據的查詢怎麼都出我們要的結果
# 會節省非常多時間在特徵工程上

# +
# 68. How to get the n’th largest value of a column when grouped by another column?
# Difficulty Level: L2
df = pd.DataFrame({'fruit': ['apple', 'banana', 'orange'] * 3,
                   'rating': np.random.rand(9),
                   'price': np.random.randint(0, 15, 9)})

df_apple = df.groupby(['fruit']).get_group('apple')
df_apple.sort_values(by='rating', ascending=False).iloc[1]



# +
# 69. How to compute grouped mean on pandas dataframe and keep the grouped column as another column (not index)?
# Difficulty Level: L1

# In df, Compute the mean price of every fruit, while keeping the fruit as another column instead of an index.

df = pd.DataFrame({'fruit': ['apple', 'banana', 'orange'] * 3,
                   'rating': np.random.rand(9),
                   'price': np.random.randint(0, 15, 9)})

df.groupby('fruit', as_index=False).mean()

# +
# 70. How to join two dataframes by 2 columns so they have only the common rows?
# Difficulty Level: L2

# Join dataframes df1 and df2 by ‘fruit-pazham’ and ‘weight-kilo’.


df1 = pd.DataFrame({'fruit': ['apple', 'banana', 'orange'] * 3,
                    'weight': ['high', 'medium', 'low'] * 3,
                    'price': np.random.randint(0, 15, 9)})

df2 = pd.DataFrame({'pazham': ['apple', 'orange', 'pine'] * 2,
                    'kilo': ['high', 'low'] * 3,
                    'price': np.random.randint(0, 15, 6)})

pd.merge(df1, df2, left_on = ['fruit','weight'],
                    right_on = ['pazham','kilo'],
                    how = 'inner')
# +
# 71. How to remove rows from a dataframe that are present in another dataframe?
# Difficulty Level: L3

# From df1, remove the rows that are present in df2. All three columns must be the same.
df1 = pd.DataFrame({'fruit': ['apple', 'orange', 'banana'] * 3,
                    'weight': ['high', 'medium', 'low'] * 3,
                    'price': np.arange(9)})

df2 = pd.DataFrame({'fruit': ['apple', 'orange', 'pine'] * 2,
                    'weight': ['high', 'medium'] * 3,
                    'price': np.arange(6)})
display(df1,
       df2)
# More readable result
mask = ~ df1.isin(df2).all(axis='columns')
df1[mask]
# +
# 72. How to get the positions where values of two columns match?
# Difficulty Level: L2
df = pd.DataFrame({'fruit1': np.random.choice(['apple', 'orange', 'banana'], 10),
                    'fruit2': np.random.choice(['apple', 'orange', 'banana'], 10)})


np.where(df.fruit1 == df.fruit2)

# Hint
# 條件篩選 - np.where

# +
# 73. How to create lags and leads of a column in a dataframe?

df = pd.DataFrame(np.random.randint(1, 100, 20).reshape(-1, 4), columns = list('abcd'))

df = df.assign(aLag1 = df.a.shift(1),
               bLead1 = df.b.shift(-1))
df

# +
# 74. How to get the frequency of unique values in the entire dataframe?
# Difficulty Level: L2

# Get the frequency of unique values in the entire dataframe df.
df = pd.DataFrame(np.random.randint(1, 10, 20).reshape(-1, 4), columns = list('abcd'))

np.unique(df.values.reshape(1,-1), return_counts=True)

# +
# 75. How to split a text column into two separate columns?
# Difficulty Level: L2

# Split the string column in df to form a dataframe with 3 columns as shown.
df = pd.DataFrame(["STD, City    State",
"33, Kolkata    West Bengal",
"44, Chennai    Tamil Nadu",
"40, Hyderabad    Telengana",
"80, Bangalore    Karnataka"], columns=['row'])
# display(df)

df_out = df.row.str.split(',|\t', expand=True)

# Make first row as header
new_header = df_out.iloc[0]
df_out = df_out[1:]
df_out.columns = new_header
df_out
# -
# # Some functionality you should know...

# +
1.
# pd.DataFrame.melt
# Person 1, 2, 3 週一至週日的某數值
# make the dataframe display with -> columns:['weekday','PersonNo','Score']
data = {'weekday': ["Monday", "Tuesday", "Wednesday", 
         "Thursday", "Friday", "Saturday", "Sunday"],
        'Person 1': [12, 6, 5, 8, 11, 6, 4],
        'Person 2': [10, 6, 11, 5, 8, 9, 12],
        'Person 3': [8, 5, 7, 3, 7, 11, 15]}
df = pd.DataFrame(data, columns=['weekday',
        'Person 1', 'Person 2', 'Person 3'])

df_result = df.melt(id_vars=['weekday'],
                    value_vars=['Person 1','Person 2','Person 3'],
                    var_name='PersonNo',value_name='Score'
                    )
df_result.head()

# when to use
# 畫圖時經常會需要先melt, 當需要的欄位不在dataframe的值中而是在columns上或是index上時(pivot-table)
# 從 pivot-table 轉回tidy datframe (unpivot)

# ref
# https://deparkes.co.uk/2016/10/28/reshape-pandas-data-with-melt/

# +
2.
# pd.DataFrame.melt
# 再一個例子
# make dataframe display like columns:['location','name','Date','Score']
data = {'location':['A','B'],
       'name':['test','foo'],
       'Jan-2010':[12,18],
       'Feb-2010':[20,20],
       'March-2010':[30,25]}
df = pd.DataFrame(data=data, columns=data.keys())

df.melt(id_vars=['location','name'],
       var_name='Date',
       value_name='Score')


# +
# 3.
# unstack
# 處理multi-index
# make the MiltiIndex series display as tidy dataframe like : columns:['number','class','value']
index = pd.MultiIndex.from_tuples([('one', 'a'), ('one', 'b'),
                                    ('two', 'a'), ('two', 'b')])
s = pd.Series(np.arange(1.0, 5.0), index=index)

def get_tidy_df(s):
    tmp = s.unstack().reset_index().rename(columns={'index':'number'})
    tmp = tmp.melt(id_vars=['number'],
                   var_name=['class'],
                   value_name='value')
    return tmp

get_tidy_df(s)


# +
# 4
# vectorlized your costum function for pandas 
# use the function without elementwise - operation
####### example
# import jieba
# result = [seg for seg in jieba.cut("我愛Python")]
# print(result)
####### Vectorlized Solution
####### This is provide extremly fast operation 
####### when you have large text need to deal with


# import jieba
# df = pd.DataFrame({'TextCol':['我愛Python','Python愛我','對我來說，R語言算什麼']})
# def segmentation(sentence : str) -> 'list':
#     '''
#     get segmentation of a sentence
#     '''
#     return [seg for seg in jieba.cut(sentence)]

# vec_segmentation = np.vectorize(segmentation, otypes=[list])
# vec_segmentation(df['TextCol'].values)
# df['Segmentation'] = vec_segmentation(df['TextCol'].values)
# df

# Hint : 使用 segmentation(df['TextCol']) 沒辦法過，series會出現無法encode
# 單純使用vectorlize也沒辦法過，錯誤訊息說我們想把一個sequence放進一個value
# 所以我們把output-type變為list，就可以被認得，如果你有一個100 million + 的dataframe需要做segmentation
# 這會比apply快上100倍
# efficiency https://engineering.upside.com/a-beginners-guide-to-optimizing-pandas-code-for-speed-c09ef2c6a4d6

# -
# 5 
# need expand data in your df into columns?
# pivot -> reset_index() ->  columns.name = None
df = pd.DataFrame({
                   'imageId':['video10_image1','video10_image1','video10_image1',
                              'video10_image1','video10_image1'],
                   'label':['nose','left eye','right eye','left ear', 'right ear'],
                   'label_x':[177.0, 179.0, 179.0, 186.0, 188.0]
                  })
pivot = df.pivot(index='imageId',columns='label', values='label_x')
pivot_getIdx = pivot.reset_index()
display(df, 
       pivot,
       pivot_getIdx)
pivot_getIdx.columns.name = None
display('the most tricky part,actually "label" is a name of coumns instead of a column!',
        pivot_getIdx)
# More Readable -> add suffix
pivotMoreReadable = df.pivot(index='imageId',columns='label', values='label_x').add_suffix('_x').reset_index()
pivotMoreReadable.columns.name = None
display(pivotMoreReadable)

# +
# 6 multiple merge
# pd.concat cannot do that!
# functional programming with pd.merge!
########### Create data ##########
FOLDER_DIR = './csvset'
for i in range(11):
    path = FOLDER_DIR + '/' + f'data_{i}.csv'
    pd.DataFrame(np.random.randint(low=5, high=100, size=(10,10))).\
    to_csv(path, index=False)
# read it
from glob import glob
DATA_PATH_LIST = glob('./csvset/*.csv')
PREFIX = 'data'
dfSet = {}
for idx, path in enumerate(DATA_PATH_LIST):
    df_name = PREFIX + f'_{idx}'
    dfSet[df_name] = pd.read_csv(path).add_prefix(f'{df_name}_').reset_index()
############ multiple merge #############
from functools import reduce
JOINKEY = 'index'
dfList = []
for _, df in dfSet.items():
    dfList.append(df)
df_all_merged = reduce(lambda left, right : pd.merge(left, right, on=JOINKEY), dfList)
display(df_all_merged.head())


# Hint, when you think about recussive solution -> functional programming might work

# -



# # pandas tricks from Kevin Markham

# +
# pandas tricks from Kevin Markham
# Does your Series contain lists of itrms?
# 1
df = pd.DataFrame({'sandwich':['PB&J','BLT','cheese'],
             'ingredients':[['peanut butter','jelly'],
                           ['bacon','lettuce','tomato'],
                           ['swiss cheese']]},
            index=['a','b','c'])

display(df)
df.explode('ingredients')
# Hint new method in 0.25
# Data Cleaning 時非常有用
# 尤其是從json格式讀取檔案時
# 同樣的方法在pd.Series當中也有


# +
# pandas tricks from Kevin Markham
# Does your Series contain comma-separation items?
# 2
df = pd.DataFrame({'sandwich':['PB&J','BLT','cheese'],
             'ingredients':['peanut butter,jelly',
                           'bacon,lettuce,tomato',
                           'swiss cheese']},
            index=['a','b','c'])

# More readable
# 使用assign
df.assign(
    ingredients = df.ingredients.str.split(',')).\
    explode('ingredients')


# +
# pandas tricks from Kevin Markham
# Does your Series contain comma-separation items?
# And you want to expand them to new columns
# 3
df = pd.DataFrame({'sandwich':['PB&J','BLT','cheese'],
             'ingredients':['peanut butter,jelly',
                           'bacon,lettuce,tomato',
                           'swiss cheese']},
            index=['a','b','c'])

# More readable
# 使用split, expand
# 使用add_prefix增加可讀性

df.ingredients.str.split(',', expand=True).\
add_prefix('ingredients_')


# +
# pandas tricks from Kevin Markham
# Check your merge dataframe keys
# 4
df1 = pd.util.testing.makeMixedDataFrame()
df2 = df1.drop([2,3], axis='rows')

pd.merge(df1, df2, how='left',indicator=True) 

# +
# pandas tricks from Kevin Markham
# 5. agg of groupby
# hint
# 使用good的方法來避免多層的multi-index，tidy-form讓後續的分析更為方便
titanic = pd.read_csv('http://bit.ly/kaggletrain')
bad_idea = titanic.groupby('Pclass').agg({'Age':['mean','max'],
                                          'Survived':['mean']})
good = titanic.groupby('Pclass').agg(ave_age=('Age','mean'),
                                    max_age=('Age','max'),
                                    survival_rate=('Survived','mean'))
def tidy_groupby_df(df):
    tidy_df =  titanic.groupby('Pclass').agg(ave_age=('Age','mean'),
                                    max_age=('Age','max'),
                                    survival_rate=('Survived','mean'))

    return tidy_df.reset_index()

tidy_df = tidy_groupby_df(titanic)
    
display(bad_idea,
       good, 
        tidy_df)


# -
# pandas tricks from Kevin Markham
# 6. read multiple csv file and keep in a dictionary
# create random data
FOLDER_DIR = './csvset'
for i in range(11):
    path = FOLDER_DIR + '/' + f'data_{i}.csv'
    pd.DataFrame(np.random.randint(low=5, high=100, size=(10,10))).\
    to_csv(path, index=False)
# read it
from glob import glob
DATA_PATH_LIST = glob('./csvset/*.csv')
PREFIX = 'data'
dfSet = {}
for idx, path in enumerate(DATA_PATH_LIST):
    df_name = PREFIX + f'_{idx}'
    dfSet[df_name] = pd.read_csv(path)
    display(df_name,
           dfSet[df_name].head(2))


# +
# pandas tricks from Kevin Markham
# 6-1 read multiple csv file and keep in a dictionary
# create random data
FOLDER_DIR = './csvset'
for i in range(11):
    path = FOLDER_DIR + '/' + f'data_{i}.csv'
    pd.DataFrame(np.random.randint(low=5, high=100, size=(10,10))).\
    to_csv(path, index=False)
# concat it
from glob import glob
DATA_PATH_LIST = glob('./csvset/*.csv')
dfList = [pd.read_csv(file) for file in DATA_PATH_LIST]

# concat
pd.concat(dfList, ignore_index=True)
# -

# pandas tricks from Kevin Markham
# Miltiple filter creteria can be hard to write and read
df = pd.read_csv('http://bit.ly/drinksbycountry')
# save as object and use reduce 
crit1 = df.continent == 'Europe'
crit2 = df.beer_servings > 200
crit3 = df.wine_servings > 200
crit4 = df.spirit_servings > 100
from functools import reduce
criteria = reduce(lambda x, y : x & y, [crit1, crit2, crit3, crit4])
df[criteria]

# +
# pandas tricks from Kevin Markham
# mash up cat and wl , read_csv skiprows, header

# step 1 (find file) ! ls ./csvset/data_0.csv
# ! cat ./csvset/data_0.csv # step 2 , take a look
# # ! wc -l ./csvset/data_0.csv # step 3 count all rows if you want to
pd.read_csv('./csvset/data_0.csv',header=0)
# -
# pandas tricks from Kevin Markham
# remove a column from a DataFrame and store it as a separate Series?
# use pop
df = pd.DataFrame([('falcon', 'bird', 389.0),
                    ('parrot', 'bird', 24.0),
                    ('lion', 'mammal', 80.5),
                    ('monkey','mammal', np.nan)],
                   columns=('name', 'class', 'max_speed'))
display(df.shape)
popped_series = df.pop('max_speed')
display(df.shape,
       popped_series.head())

# +
# pandas tricks from Kevin Markham
# Do you need to build a DataFrame from multiple files,
# but also keep track of which row came from which file?

from glob import glob
DATA_PATH_LIST = glob('./csvset/*.csv')
print(DATA_PATH_LIST)
# use generator expression
csv_geberator = (pd.read_csv(file).assign(file_name = file)
                 for file in DATA_PATH_LIST)

# concat
pd.concat(csv_geberator, ignore_index=True)
# +
# pandas tricks from Kevin Markham
# create a training set dtype_validator?
# 1 Create a CSV of column names & dtypes
# 2 Read it into a DataFrame, then convery it into a dictionary
# Use the dictionary to specify dtypes of the data

# When to use? 
# you have 50+ columns from different source and device....
# Why we use a pickle for dictionary? 
# we need to pick one readable for human, makes more maintainable
# we pick json instead of csv, because json is smaller
# hints : json only takes double quote, single quote is invalid...
import json
with open('validator_reference/column_dtypes.json', 'r') as file:
    dtypes = json.loads(file.read())
df = pd.read_csv('http://bit.ly/drinksbycountry', dtype=dtypes)
display(df.dtypes)
# +
# pandas tricks from Kevin Markham
# read  big csv into data-frame?
# Randomly sample the dataset  *during file reading* by *skiprows*

# create the data if not exist
import os 
if 'huge_dataset.csv' not in os.listdir('./csvset'):
    data = np.random.uniform(0, 1, (100000, 100))
    pd.DataFrame(data).to_csv('./csvset/huge_dataset.csv', index=False)


# neat
print('we have 100000 raws huge data!')
df = pd.read_csv('./csvset/huge_dataset.csv', 
                skiprows = lambda x : x > 0 and np.random.rand() > 0.01)

print('sample data with fraction 1 %',df.shape)

# clearn the data
import subprocess

delete_file = 'rm ./csvset/huge_dataset.csv'
subprocess.run(delete_file, shell=True, check=True)
# -


# ## How it works?
# * `skiprows` accepts a function that is evaluated against the integer index
# * `x > 0`ensures that the header row is **not** skipped
# * `np.random.rand() > 0.01` return **True** 99% of the time, thus skip 99% of the rows
# ### different with sample
# * `sample` need whole dataset in dataframe!

# +
# pandas tricks from Kevin Markham
# Want to convert "year" and "day of year" into a single datetime column? 📆
# 1. Combine them into one number
# 2. Convert to datetime and specify its format

df = pd.DataFrame({'year':[2019,2019,2020],
                  'day_of_yesr':[350, 365, 1]})

# step 1 
df['combined'] = df['year'] * 1000 + df['day_of_yesr']

# step 2
df['date'] = pd.to_datetime(df['combined'], format='%Y%j')
df
# -
# # pandas trick from LeeMeng


# +
# 找出符合特定字串的樣本

df = pd.read_csv('http://bit.ly/kaggletrain')
df[df.Name.str.contains("Mr\.")].head(5)



# +
# 用正規表達式來選data
# 尤其是時間序列

df_date = pd.util.testing.makeTimeDataFrame(freq='7D')
df_date.head(10)
# -


# a more readable way to filter by index with alomst any operatioms
df_date.filter(regex="2000-02.*", axis=0)

# +
# 選取從某段時間開始的區間樣本
#        df_date.between_time(start_time=datetime(2000, 2, 5),
#                           end_time=datetime(2000, 3, 19))
# will not work
# do a old school way
from datetime import datetime
def between(df : pd.DataFrame, 
            start_date : datetime,
            end_date : datetime,
           *args) -> pd.DataFrame:
    assert isinstance(df.index, pd.DatetimeIndex), 'index is not DatetimeIndex format'
    mask = (df.index > start_date) & (df.index <= end_date)
    return df[mask]
    

display(df_date.first('3W'),
       between(df_date,start_date=datetime(2000,1, 15),
              end_date=datetime(2000, 3, 10)))

# +
# for loop but want multiple row
df_city = pd.DataFrame({
    'state': ['密蘇里州', '亞利桑那州', '肯塔基州', '紐約州'],
    'city': ['堪薩斯城', '鳳凰城', '路易維爾', '紐約市']
})

for row in df_city.itertuples(name='City'):
    print(f'{row.city}是{row.state}裡頭的一個城市')

    
# itertuples is exactly namedtuple
from collections import namedtuple

City = namedtuple('City', ['Index', 'state', 'city'])
c = City(3, '紐約州', '紐約市')
c == row

# +
# swifter, star 1.2k, then might be well tested
# https://github.com/jmcarpenter2/swifter
# use swifter auto vectorlization of your code
# but keep your ability to write vectorlize code :P
# actually apply support numba backend now!

import swifter
df = pd.DataFrame(pd.np.random.rand(1000000, 1), columns=['x'])

# %timeit -n 10 df['x2'] = df['x'].apply(lambda x: x**2)
# %timeit -n 10 df['x2'] = df['x'].swifter.apply(lambda x: x**2)


# -


