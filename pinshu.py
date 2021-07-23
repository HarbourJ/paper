# -*- coding: utf-8 -*-
# @Time    : 2020/12/1 13:34
# @Author  : HEJEN
# @File    : pinshu.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=True
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import model_function

# 导入FSC检查原始数据集
raw_dataset = pd.read_csv("2017_FSC_data.csv", na_values="?", comment='\t',
                          sep=",", dtype={'缺陷代码':str},skipinitialspace=True)
raw_dataset.drop(['船舶识别号','初查/复查','安检类型','处理意见说明','缺陷描述'],axis=1,inplace=True)
dataset = raw_dataset.copy()
# print(dataset)

"""
将缺陷代码转化为大类并统计
"""

# 缺陷代码转化为大类并统计
Defect_Code = dataset['缺陷代码'].astype('str').apply(lambda s: s[:-2] + '00')
a = Defect_Code.value_counts()
# print('缺陷大类及频数统计:\n', a)
a.drop(['n00'], inplace=True)

x=[]#定义两个列表
y=[]
list=sorted(a.items(),key=lambda item:item[1],reverse=True)#得到的是一个list,list中的元素是tuple
for i in list:
    x.append(i[0])
    y.append(i[1])

model_function.plot_statistical_chart(x, y)