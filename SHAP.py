# -*- coding: utf-8 -*-
# @Time    : 2020/12/9 13:55
# @Author  : HEJEN
# @File    : SHAP.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier as XGBC
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
import model_function

df = pd.read_csv('2017_FSC_data_done.csv')
data = df.copy()
X = data.drop('Detained', 1)
y = data['Detained']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=10) #split the data
# XGB
model = XGBC(n_estimators=110, max_depth=3)
model.fit(X_train, y_train)
cols = X.columns.values
# print(cols)
# model_function.plot_feature_importance(cols, model)

# model_function.plot_PermutationImportance(X_test, y_test, model)

model_function.plot_shap_values(X_train, model)


# model_function.plot_shap_interaction_values(X_train, model)




























