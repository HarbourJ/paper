from collections import Counter
#使用imlbearn库中上采样方法中的SMOTE接口
from imblearn.over_sampling import SMOTE
#可通过sampling_strategy参数指定对应类别要生成的数据的数量
import pandas as pd
from sklearn.model_selection import train_test_split #for data splitting
from xgboost import XGBClassifier as XGBC
from sklearn.ensemble import RandomForestClassifier # RandomForest
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, recall_score, accuracy_score
from sklearn.metrics import *
import warnings
warnings.filterwarnings('ignore')
import model_function

"""
读取数据集
"""
df = pd.read_csv('2017_FSC_data_done.csv')
data = df.copy()
X = data.drop('Detained', 1)
y = data['Detained']

"""
SMOTE过采样
"""
smo = SMOTE(sampling_strategy={1: 4005},random_state=42)
#生成0和1比例为3比1的数据样本
X_smo, y_smo = smo.fit_sample(X, y)
((y_smo == 1).sum() / y_smo.shape[0])
print(Counter(y_smo))


"""
划分数据集
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=10) #split the data
X_smo_train, X_smo_test, y_smo_train, y_smo_test = train_test_split(X_smo, y_smo, test_size = 0.3, random_state=10)


"""
设定算法参数、训练模型
"""
# XGB
param_dist = {'objective':'binary:logistic',
              'n_estimators':1000,
              'max_depth':30,
              'learning_rate':0.01,
              'booster':'gbtree',
              'min_child_weight': 5,
              'colsample_btree':0.8,
              'colsample_bylevel':0.8,
              }
model = XGBC(**param_dist)
model.fit(X_train, y_train)
# sklearn接口
model1 = XGBC(n_estimators=110, max_depth=3)
model1.fit(X_train, y_train)

# SMO_XGB
model_smo = XGBC(**param_dist)
model_smo.fit(X_smo_train, y_smo_train)
# RF
model_rf = RandomForestClassifier(n_estimators=10,max_depth=4) # random_state=0
model_rf.fit(X_train, y_train)
# SVM
model_svm = SVC(C=2, kernel='rbf', probability=True)
model_svm.fit(X_train, y_train)
# Logistic
model_log = LogisticRegression(C = 10, penalty='l2', solver='liblinear')
model_log.fit(X_train, y_train)


"""
预测值 pre
"""
# XGB
y_pred = model.predict(X_test)
y_score = model.predict_proba(X_test)
y_pred_bin = model.predict(X_test)
# XGB sklearn接口
y_pred1 = model1.predict(X_test)
y_score1 = model1.predict_proba(X_test)
y_pred_bin1 = model1.predict(X_test)

# SMO_XGB
y_smo_pred = model_smo.predict(X_smo_test)
y_smo_score = model_smo.predict_proba(X_smo_test)
y_smo_pred_bin = model_smo.predict(X_smo_test)
# RF
y_rf_pred = model_rf.predict(X_test)
y_rf_score = model_rf.predict_proba(X_test)
y_rf_pred_bin = model_rf.predict(X_test)
# SVM
y_svm_pred = model_svm.predict(X_test)
y_svm_score = model_svm.predict_proba(X_test)
y_svm_pred_bin = model_svm.predict(X_test)
# Logistic
y_log_pred = model_log.predict(X_test)
y_log_score = model_log.predict_proba(X_test)
y_log_pred_bin = model_log.predict(X_test)


"""
FPR、TPR
"""
fpr, tpr, thresholds = roc_curve(y_test, y_score[:, 1])  # XGB
fpr1, tpr1, thresholds1 = roc_curve(y_test, y_score1[:, 1])  # XGB sklearn接口
fpr_smo, tpr_smo, thresholds_smo = roc_curve(y_smo_test, y_smo_score[:, 1])  # SMO_XGB
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_rf_score[:, 1])  # RF
fpr_svm, tpr_svm, thresholds_svm = roc_curve(y_test, y_svm_score[:, 1])  # SVM
fpr_log, tpr_log, thresholds_log = roc_curve(y_test, y_log_score[:, 1])  # Logistic

precision1, recall1, thr = precision_recall_curve(y_test, model.predict_proba(X_test)[:, 1])
precision2, recall2, thr = precision_recall_curve(y_smo_test, model_smo.predict_proba(X_smo_test)[:, 1])
precision3, recall3, thr = precision_recall_curve(y_test, model_rf.predict_proba(X_test)[:, 1])
precision4, recall4, thr = precision_recall_curve(y_test, model_svm.predict_proba(X_test)[:, 1])
precision5, recall5, thr = precision_recall_curve(y_test, model_log.predict_proba(X_test)[:, 1])


# print('XGB_accuracy: %.3f'%model.score(X_test,y_test))
# # print('accuracy_: %s'%accuracy_score(y_test,y_pred))
# print('SMO_XGB_accuracy: %.3f'%model_smo.score(X_smo_test,y_smo_test))
# print('RF_accuracy: %.3f'%model_rf.score(X_test,y_test))
# print('SVM_accuracy: %.3f'%model_svm.score(X_test,y_test))
# print('Logistic_accuracy: %.3f'%model_log.score(X_test,y_test))
# print('-'*20)
# print('XGB_recall: %s'%recall_score(y_test,y_pred))
# print('SMO_XGB_recall: %s'%recall_score(y_smo_test,y_smo_pred))
# print('RF_recall: %s'%recall_score(y_test,y_rf_pred))
# print('SVM_recall: %s'%recall_score(y_test,y_svm_pred))
# print('Logistic_recall: %s'%recall_score(y_test,y_log_pred))
# print('-'*20)
# print('XGB_AUC: %s'%roc_auc_score(y_test,model.predict_proba(X_test)[:,1]))
# print('SMO_XGB_AUC: %s'%roc_auc_score(y_smo_test,model_smo.predict_proba(X_smo_test)[:,1]))
# print('RF_AUC: %s'%roc_auc_score(y_test,model_rf.predict_proba(X_test)[:,1]))
# print('SVM_AUC: %s'%roc_auc_score(y_test,model_svm.predict_proba(X_test)[:,1]))
# print('Logistic_AUC: %s'%roc_auc_score(y_test,model_log.predict_proba(X_test)[:,1]))



# model_function.plot_roc_curve(fpr, tpr ,fpr_smo, tpr_smo,fpr_rf,tpr_rf,fpr_svm,tpr_svm,fpr_log,tpr_log)

# print ("================================")
# print('XGBoost AUC: %s ' %auc(fpr ,tpr))
# print('SMOTE-XGB AUC: %s ' %auc(fpr_smo ,tpr_smo))
# print('RF_AUC: %s ' %auc(fpr_rf, tpr_rf))
# print('SVM_AUC: %s ' %auc(fpr_svm, tpr_svm))
# print('Logistic_AUC: %s ' %auc(fpr_log, tpr_log))

# model_function.Loss_curve(X_train,y_train,X_test,y_test)

# model_function.plot_roc_curve(fpr, tpr, fpr1, tpr1, fpr_smo, tpr_smo, fpr_rf, tpr_rf, fpr_svm, tpr_svm, fpr_log, tpr_log)

# model_function.plot_pr_curve(precision1, recall1, precision2, recall2, precision3, recall3, precision4, recall4, precision5, recall5)

# model_function.plot_confusion_matrix(y_test, y_rf_pred)
# model_function.plot_confusion_matrix(y_test, y_svm_pred)
# model_function.plot_confusion_matrix(y_test, y_log_pred)