from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
import numpy as np
import  matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn.externals import joblib


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

# date_form函数
def date_format(dt):
    import time
    t = time.strptime(' '.join(dt), '%d/%m/%Y %H:%M:%S')
    return (t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec)

# lambda和普通的函数相比，就是省去了函数名称而已，同时这样的匿名函数，又不能共享在别的地方调用。
X = X.apply(lambda x: pd.Series(date_format(x)), axis=1)
Y = datas['Global_active_power']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# 处理数据

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# print(X_train.describe().T)
# 训练模型
lr = LinearRegression()
lr.fit(X_train, Y_train) #训练模型

# 模型校验
y_predict = lr.predict(X_test)  #预测结果

# print("训练R2", lr.score(X_train, Y_train))
# print("测试R2:",lr.score(X_test, Y_test))

# 平均方差
mse = np.average((y_predict-Y_test)**2)
rmse = np.sqrt(mse)
# print('rmse:', rmse)

# 模型保存
joblib.dump(ss, 'data_ss.model')
joblib.dump(lr, 'data_lr.model')

# 利用保存好的模型训练数据
ss1 = joblib.load('data_ss.model')
lr1 = joblib.load('data_lr.model')

'''
# 使用加载的模型进行预测
data1 = [[2006, 12, 17, 12, 25, 0]]
data1 = ss1.transform(data1)
lr1.predict(data1)
'''

# 绘图，时间和功率之间的关系
t = np.arange(len(X_test))
plt.figure(facecolor='w')
plt.plot(t, Y_test, 'r-', linewidth=2, label='real value')
plt.plot(t, y_predict, 'g-', linewidth=2, label='predict value')
plt.legend(loc = 'upper left')
plt.title('liner regression for time and power', fontsize=20)
plt.grid(b=True)
plt.show()

