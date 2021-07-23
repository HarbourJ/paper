from sklearn.metrics import *
import matplotlib.pyplot as plt
from xgboost import XGBClassifier as XGBC
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=True
import warnings
warnings.filterwarnings('ignore')

def plot_statistical_chart(x, y):
    """
    绘制缺陷代码大类频数统计图
    """

    plt.figure(figsize=(12, 5))  # dpi=130
    # plt.title("Frequency distribution of deficiency code", fontsize=18)
    # plt.xticks([index for index in range(len(y))], x)
    plt.xlabel('Deficiency code(major category)', fontsize=16)
    plt.ylabel('Frequency', fontsize=16)
    plt.grid(axis='y', linestyle=':')
    plt.bar(x, y, width=0.6,
            color=['c','darkorange','darkgreen','RoyalBlue','darkmagenta'],
            tick_label=x)
    plt.xlim(-0.6, 22)
    plt.ylim(0, 17000)

    # 给图形添加数据标签
    for __x, __y in zip(range(len(x)), y):
        plt.text(__x, __y, '%.0f' % __y, ha='center', va='bottom', size=10)
    plt.tight_layout()
    # 保存频数图
    # plt.savefig("Pinshutud.svg")
    ps_fig = plt.gcf()
    # ps_fig.savefig('Pinshutu.svg', format='svg', dpi=1000)
    plt.show()

"""
绘制roc曲线
"""
def plot_roc_curve(fpr,tpr,fpr1,tpr1,fpr_smo,tpr_smo,fpr_rf,tpr_rf,fpr_svm,tpr_svm,fpr_log,tpr_log):
    #     plt.figure()
    plt.figure('ROC curve', figsize=(10, 9),
               dpi=150,
               )  # dpi=180
    lw = 2
    roc_auc = auc(fpr, tpr)
    roc_auc1 = auc(fpr1, tpr1)
    roc_auc_smo = auc(fpr_smo, tpr_smo)
    roc_auc_rf = auc(fpr_rf, tpr_rf)
    roc_auc_svm = auc(fpr_svm, tpr_svm)
    roc_auc_log = auc(fpr_log, tpr_log)
    plt.plot(fpr, tpr, color='darkorange', lw=lw,
             label='XGBoost (AUC = %0.3f)' % roc_auc)
    # plt.plot(fpr1, tpr1, color='darkorchid', lw=lw,
    #          label='XGBoost_sklearn (AUC = %0.3f)' % roc_auc)
    plt.plot(fpr_smo, tpr_smo, color='RoyalBlue', lw=lw,
             label='SMOTE-XGB-SD (AUC = %0.3f)' % roc_auc_smo)
    plt.plot(fpr_rf, tpr_rf, color='darkmagenta', lw=lw,
             label='RF (AUC = %0.3f)' % roc_auc_rf)
    plt.plot(fpr_svm, tpr_svm, color='darkcyan', lw=lw,
             label='SVM (AUC = %0.3f)' % roc_auc_svm)
    plt.plot(fpr_log, tpr_log, color='darkgreen', lw=lw,
             label='LR (AUC = %0.3f)' % roc_auc_log)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, ls='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate',fontsize=14)
    plt.ylabel('True Positive Rate',fontsize=14)
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.grid(
        linestyle=':'
    )
    roc_fig = plt.gcf()
    roc_fig.savefig('ROC curve.svg', format='svg', dpi=300)
    plt.show()
def plot_pr_curve(precision1, recall1, precision2, recall2,
                  precision3, recall3, precision4, recall4, precision5, recall5):

    from sklearn.metrics import precision_recall_curve
    plt.figure(figsize=(12, 12))
    plt.plot(recall1, precision1, c="darkorange", label="XGB")
    plt.plot(recall2, precision2, c="RoyalBlue", label="SMO-XGB-SD")
    plt.plot(recall3, precision3, c="darkmagenta", label="RF")
    plt.plot(recall4, precision4, c="darkcyan", label="SVM")
    plt.plot(recall5, precision5, c="darkgreen", label="LR")
    plt.xlabel('recall', fontsize=14)
    plt.ylabel('precision', fontsize=14)
    plt.xlim([0.2, 1.0])
    plt.ylim([0, 1.0])
    plt.legend()
    roc_fig = plt.gcf()
    roc_fig.savefig('PR curve.svg', format='svg', dpi=300)
    plt.show()


def plot_Loss_curve(X_train, y_train, X_test, y_test):
    print()
    print(format('How to visualise XGBoost model with learning curves','*^82'))

    # load libraries
    plt.style.use('ggplot')

    # fit model no training data
    model = XGBC()
    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(X_train, y_train, early_stopping_rounds=10, eval_metric=["error", "logloss"],
              eval_set=eval_set, verbose=False)

    # make predictions for test data
    y_pred = model.predict(X_test)

    # evaluate predictions
    accuracy = model.score(X_test, y_test)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    # retrieve performance metrics
    results = model.evals_result()
    epochs = len(results['validation_0']['error'])
    x_axis = range(0, epochs)

    # plot log loss
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(x_axis, results['validation_0']['logloss'], label='Train', marker='o')
    ax.plot(x_axis, results['validation_1']['logloss'], label='Test', marker='x')
    ax.legend()
    plt.ylabel('Log Loss')
    plt.title('XGBoost Log Loss')
    plt.show()

    # plot classification error
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_axis, results['validation_0']['error'], label='Train')
    ax.plot(x_axis, results['validation_1']['error'], label='Test')
    ax.legend()
    plt.ylabel('Classification Error')
    plt.title('XGBoost Classification Error')
    plt.show()

    # plot xgb auc
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['auc'], label='Train')
    ax.plot(x_axis, results['validation_1']['auc'], label='Test')
    ax.legend()
    plt.ylabel('Xgboost Auc')
    plt.title('Xgboost Auc')
    plt.show()

def plot_feature_importance(cols, model):
    plt.figure(figsize=(15, 5))
    plt.bar(range(len(cols)), model.feature_importances_)
    plt.xticks(range(len(cols)), cols, rotation=-45, fontsize=8)
    plt.title('Feature importance', fontsize=14)
    plt.show()

def plot_PermutationImportance(X_test, y_test, model):
    import eli5
    from eli5.sklearn import PermutationImportance

    # first_model是已经训练好的模型
    perm = PermutationImportance(model, random_state=1).fit(X_test, y_test)
    # 显示结果
    eli5.show_weights(perm, feature_names = X_test.columns.tolist())



def plot_shap_values(X_train, model):
    import shap
    plt.style.use('seaborn')
    shap_values = shap.TreeExplainer(model).shap_values(X_train)

    plt.figure(dpi=500)
    fig=plt.gcf()

    shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)

    fig.set_facecolor('white')
    # fig.savefig('Figure 111.svg', format='svg', dpi=300)
    fig.set_facecolor('white')  # 把背景颜色设置为白色
    fig.savefig('filename.png', bbox_inches='tight', dpi=500)  # 保存图片
    # shap.summary_plot(shap_values, X_train)
    # shap.dependence_plot('Defect_Nums', shap_values, X_train)
    # plt.show()

"""
对多个变量的交互进行分析
"""
def plot_shap_interaction_values(X_train, model):
    import shap
    plt.style.use('seaborn')
    shap_interaction_values = shap.TreeExplainer(model).shap_interaction_values(X_train)
    shap.summary_plot(shap_interaction_values, X_train, max_display=5)
    plt.show()

def plot_show_weights(X_test, y_test, model):
    import eli5
    from eli5.sklearn import PermutationImportance

    perm = PermutationImportance(model, random_state=1).fit(X_test, y_test)
    # 显示结果
    eli5.show_weights(perm, feature_names=X_test.columns.tolist())


def plot_confusion_matrix(y_test, y_pred):
    # 混淆矩阵可视化
    import seaborn as sns
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    # plt.figure(figsize=(5,5), dpi=200)
    sns.heatmap(cm, linewidths=0.1,
                vmax=100.0,
                square=True,
                linecolor='white',
                annot=True,
                cmap="Blues")
    plt.show()

