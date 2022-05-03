import time
import pandas as pd
import xlwt

#原始excel文件名


#读取excel中所有数据


#选取数据中需要的部分，先是列，后是行


def train():  # 训练数据集
    fr1 =  pd.read_excel(io='Test.xlsx',sheet_name='Sheet1')
    data = fr1.iloc[:, 1:]
    dataMat=data.values.tolist()
    print(dataMat)

    fr2 =  pd.read_excel(io='Test.xlsx',sheet_name='Sheet1')
    data2 = fr2.iloc[:, 0]
    labelMat0=data2.values.tolist()
    labelMat=[]
    for i in labelMat0:
        labelMat.append(i)
    print(labelMat)
    return dataMat, labelMat


def test():  # 测试数据集
    fr1 =  pd.read_excel(io='Test.xlsx',sheet_name='Sheet2')
    data = fr1.iloc[:, 1:]
    dataMat=data.values.tolist()
    print(dataMat)

    fr2 =  pd.read_excel(io='Test.xlsx',sheet_name='Sheet2')
    data2 = fr2.iloc[:, 0]
    labelMat0=data2.values.tolist()
    labelMat=[]
    for i in labelMat0:
        labelMat.append(i)
    print(labelMat)
    return dataMat, labelMat


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm

predictor = svm.SVC(gamma='scale', C=1.0, decision_function_shape='ovr', kernel='poly', degree=20)
'''gamma:内核系数
c：错误项的惩罚参数
kernel：核函数
degree：多项式poly函数的阶数
'''
# 训练

from sklearn import model_selection
import pandas as pd
# 读入数据
df = pd.read_excel(r'Test.xlsx')
df.head()
# X = df[['q',	'w',	'e',	'r',	't',	'y',	'u',	'i',	'o',	'p',	'a',	's',	'd',	'f',	'g',	'h',	'j',	'k'	,'l',	'z']]
X = df[['q',	'w',	'e',	'r',	't',	'y',	'u',	'i',	'o',	'p',	'a',	's',	'd',	'f',	'g']]
y = df['mm']

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.1, random_state = 4321)
# x_train, y_train = train()
x_train=X_train.values.tolist()
y_train=y_train.values.tolist()
predictor.fit(x_train, y_train)
# 预测
tp = 0
fp = 0
p = 0
result = predictor.predict(X_test)
result=list(result)
y_test=list(y_test)
for i in range(len(y_test)):
    if y_test[i]==result[i]:
        p=p+1# 评估
print('test_size: 0.1')
print('random_state: 4321')
print("准确率:", 100 * p / len(y_test), "%")
