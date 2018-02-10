from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import  matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
import time

# 加载数据
dataPath = 'data/household_power_consumption_1000.txt'
df = pd.read_csv(dataPath, sep=';', low_memory=False)

'''
# 头部信息
df.head()
# 格式信息
df.info()
# 统计信息
df.describe()
'''
new_df = df.replace('?', np.nan)
datas = new_df.dropna(axis=0, how='any')

'''
# loc——通过行标签索引行数据 
# iloc——通过行号索引行数据 
# ix——通过行标签或者行号索引行数据（基于loc和iloc 的混合） 
'''
X = datas.iloc[:, 0:2]

'''
# lambda和普通的函数相比，就是省去了函数名称而已，同时这样的匿名函数，又不能共享在别的地方调用。
'''

# date_form函数
def date_format(dt):
    import time
    t = time.strptime(' '.join(dt), '%d/%m/%Y %H:%M:%S')
    return (t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec)

X = X.apply(lambda x: pd.Series(date_format(x)), axis=1)
Y = datas['Global_active_power']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

print(X_train.describe().T)





