
import numpy as np

import  pandas as pd

import matplotlib.pyplot as plt


values_dict = {
    'Acc': [0.978, 0.993, 0.976, 0.976, 0.971],
    'P': [0.78, 0.98, 0.98, 0.91, 0.52],
    'Recall': [0.35, 0.83, 0.17, 0.27, 0.11],
    'F1 score': [0.48, 0.88, 0.28, 0.41, 0.18]
}
index = ['XGBoost', 'SMO-XGB-SD', 'RF', 'SVM', 'LR']
df_merics = pd.DataFrame(values_dict, index=index)
df_merics = df_merics.stack().unstack(0)
print(df_merics)

# plt.figure(figsize=(12,8))
tick_label = ['Acc', 'P', 'Recall', 'F1 score']
pos = list(range(len(df_merics)))
width = 0.17
# plt.barh([p - 2*width for p in pos], df_merics['XGBoost'], width, color='r', label="XGBoost")
# plt.barh([p - width for p in pos], df_merics['SMO-XGB-SD'], width, color='g', label="SMO-XGB-SD")
# plt.barh([p for p in pos], df_merics['RF'], width, color='b', label="RF")
# plt.barh([p + width for p in pos], df_merics['SVM'], width, color='r', label="SVM")
# plt.barh([p + 2*width for p in pos], df_merics['LR'], width, color='g', label="LR")
# plt.show()

plt.bar([p - 2*width for p in pos], df_merics['XGBoost'], width, color='dodgerblue',  label="XGBoost")
plt.bar([p - width for p in pos], df_merics['SMO-XGB-SD'], width, color='darkorange', label="SMO-XGB-SD")
plt.bar([p for p in pos], df_merics['RF'], width, color='darkorchid', label="RF")
plt.bar([p + width for p in pos], df_merics['SVM'], width, color='darkgreen', label="SVM")
plt.bar([p + 2*width for p in pos], df_merics['LR'], width, color='deepskyblue', label="LR")
# plt.xlim(0.1, 1.0, 0.1)
plt.ylim(0.10, 1.0, 0.10)
plt.xticks([p for p in pos], tick_label, fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
plt.tight_layout()
plt.show()