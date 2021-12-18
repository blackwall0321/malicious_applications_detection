import pandas as pd
import numpy as np
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler# 逻辑回归
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier# 随机森林
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier# K近邻法
from sklearn.tree import DecisionTreeClassifier#决策树
from sklearn import svm#支持向量机
from sklearn import naive_bayes#朴素贝叶斯
from sklearn.neural_network import MLPClassifier#神经网络
import os,sys
from sklearn.metrics import precision_score, recall_score,accuracy_score,f1_score,roc_curve,auc
import matplotlib.pyplot as plt
import xgboost_test as xgb
picture_path = sys.path[0]+'\\picture\\'

def Logistic_sklearn(data):
    AUC = []
    ACC = []
    PRE = []
    REC = []
    F1 = []
    fpr = []
    tpr = []
    num = [45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
    for i in range(len(num)):
        AUC_temp, ACC_temp, PRE_temp, REC_temp, F1_temp, fpr_temp, tpr_temp = Logistic_sklearn_son(data, num[i])
        AUC.append(AUC_temp)
        ACC.append(ACC_temp)
        PRE.append(PRE_temp)
        REC.append(REC_temp)
        F1.append(F1_temp)
        fpr.append(fpr_temp)
        tpr.append(tpr_temp)


    temp = np.argmax(ACC)#取出ACC中最大值对应的索引
    print('逻辑回归 ACC = ', ACC[temp],'AUC = ', AUC[temp],  'PRE = ', PRE[temp], 'REC = ', REC[temp], 'f1 = ', F1[temp])
    print(ACC)
    return fpr[temp],tpr[temp],AUC[temp],ACC
def Logistic_sklearn_son(data,num):
    columns_name = list(data.columns)
    columns_name.remove('safe_or_bad')
    columns_name.remove('app_name')
    X_train, X_test, y_train, y_test = train_test_split(data[columns_name], data['safe_or_bad'], test_size=num*0.01)
    # print(y_test)
    # 使用逻辑回归
    lr = LogisticRegression()
    lr.fit(X_train, y_train)  # 评分
    predictions = lr.predict(X_test)

    y_probs = lr.predict_proba(X_test)[:, 1]  # 模型的预测得分
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    AUC = auc(fpr, tpr)  # auc为Roc曲线下的面积
    return  AUC ,accuracy_score(y_test, predictions),precision_score(y_test, predictions), recall_score(y_test, predictions), f1_score(y_test, predictions),fpr,tpr

def Random_sklearn(data):
    AUC = []
    ACC = []
    PRE = []
    REC = []
    F1 = []
    fpr = []
    tpr = []
    num = [45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
    for i in range(len(num)):
        AUC_temp, ACC_temp, PRE_temp, REC_temp, F1_temp, fpr_temp, tpr_temp = Random_sklearn_son(data, num[i])
        AUC.append(AUC_temp)
        ACC.append(ACC_temp)
        PRE.append(PRE_temp)
        REC.append(REC_temp)
        F1.append(F1_temp)
        fpr.append(fpr_temp)
        tpr.append(tpr_temp)
    temp = np.argmax(ACC)
    print('随机森林 ACC = ', ACC[temp], 'AUC = ', AUC[temp], 'PRE = ', PRE[temp], 'REC = ', REC[temp], 'f1 = ', F1[temp])
    print(ACC)
    return fpr[temp], tpr[temp], AUC[temp]
def Random_sklearn_son(data,num):
    columns_name = list(data.columns)
    columns_name.remove('safe_or_bad')
    columns_name.remove('app_name')  # 选择样本特征和类别输出
    X_train, X_test, y_train, y_test = train_test_split(data[columns_name], data['safe_or_bad'], test_size=num * 0.01)
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)  # 准确率
    predictions = rf.predict(X_test)
    y_probs = rf.predict_proba(X_test)[:, 1]  # 模型的预测得分
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    AUC = auc(fpr, tpr)  # auc为Roc曲线下的面积
    return AUC, accuracy_score(y_test, predictions), precision_score(y_test, predictions), recall_score(y_test,predictions), f1_score(y_test, predictions), fpr, tpr

def KN_sklearn(data):
    AUC = []
    ACC = []
    PRE = []
    REC = []
    F1 = []
    fpr = []
    tpr = []
    num = [45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
    for i in range(len(num)):
        AUC_temp, ACC_temp, PRE_temp, REC_temp, F1_temp, fpr_temp, tpr_temp = KN_sklearn_son(data, num[i])
        AUC.append(AUC_temp)
        ACC.append(ACC_temp)
        PRE.append(PRE_temp)
        REC.append(REC_temp)
        F1.append(F1_temp)
        fpr.append(fpr_temp)
        tpr.append(tpr_temp)

    temp = np.argmax(ACC)
    print('K近邻法 ACC = ', ACC[temp], 'AUC = ', AUC[temp], 'PRE = ', PRE[temp], 'REC = ', REC[temp], 'f1 = ', F1[temp])
    print(ACC)
    return fpr[temp], tpr[temp], AUC[temp]
def KN_sklearn_son(data,num):
    columns_name = list(data.columns)
    columns_name.remove('safe_or_bad')
    columns_name.remove('app_name')
    # 随机采样构建训练测试集合
    X_train, X_test, y_train, y_test = train_test_split(data[columns_name], data['safe_or_bad'], test_size=num *0.01)
    # 使用KNN分类器
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)

    y_probs = knn.predict_proba(X_test)[:, 1]  # 模型的预测得分
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    AUC = auc(fpr, tpr)  # auc为Roc曲线下的面积
    return AUC, accuracy_score(y_test, predictions), precision_score(y_test, predictions), recall_score(y_test,predictions), f1_score(y_test, predictions),fpr,tpr

def Decision_sklearn(data):
    AUC = []
    ACC = []
    PRE = []
    REC = []
    F1 = []
    fpr = []
    tpr = []
    num = [45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
    for i in range(len(num)):
        AUC_temp, ACC_temp, PRE_temp, REC_temp, F1_temp, fpr_temp, tpr_temp = Decision_sklearn_son(data, num[i])
        AUC.append(AUC_temp)
        ACC.append(ACC_temp)
        PRE.append(PRE_temp)
        REC.append(REC_temp)
        F1.append(F1_temp)
        fpr.append(fpr_temp)
        tpr.append(tpr_temp)
    temp = np.argmax(ACC)
    print('决策树 ACC = ', ACC[temp], 'AUC = ', AUC[temp], 'PRE = ', PRE[temp], 'REC = ', REC[temp], 'f1 = ', F1[temp])
    print(ACC)
    return fpr[temp], tpr[temp], AUC[temp]
def Decision_sklearn_son(data,num):
    columns_name = list(data.columns)
    columns_name.remove('safe_or_bad')
    columns_name.remove('app_name')
    # 随机采样构建训练测试集合
    X_train, X_test, y_train, y_test = train_test_split(data[columns_name], data['safe_or_bad'], test_size=num *0.01)
    # 标准化数据
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)
    # 初始化
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    predictions = dtc.predict(X_test)

    y_probs = dtc.predict_proba(X_test)[:, 1]  # 模型的预测得分
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    AUC = auc(fpr, tpr)  # auc为Roc曲线下的面积
    return AUC, accuracy_score(y_test, predictions), precision_score(y_test, predictions), recall_score(y_test,predictions), f1_score(y_test, predictions),fpr,tpr

def svm_sklearn(data):
    AUC = []
    ACC = []
    PRE = []
    REC = []
    F1 = []
    fpr = []
    tpr = []
    num = [45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
    for i in range(len(num)):
        AUC_temp, ACC_temp, PRE_temp, REC_temp, F1_temp, fpr_temp, tpr_temp = svm_sklearn_son(data, num[i])
        AUC.append(AUC_temp)
        ACC.append(ACC_temp)
        PRE.append(PRE_temp)
        REC.append(REC_temp)
        F1.append(F1_temp)
        fpr.append(fpr_temp)
        tpr.append(tpr_temp)

    temp = np.argmax(ACC)

    print('支持向量机 ACC = ', ACC[temp], 'AUC = ', AUC[temp], 'PRE = ', PRE[temp], 'REC = ', REC[temp], 'f1 = ', F1[temp])
    print(ACC)
    return fpr[temp], tpr[temp], AUC[temp]
def svm_sklearn_son(data,num):
    columns_name = list(data.columns)
    columns_name.remove('safe_or_bad')
    columns_name.remove('app_name')
    X_train, X_test, y_train, y_test = train_test_split(data[columns_name], data['safe_or_bad'], test_size=num *0.01)
    # 标准化数据
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)
    clf = svm.SVC(probability=True)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    y_probs = clf.predict_proba(X_test)[:, 1]  # 模型的预测得分
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    AUC = auc(fpr, tpr)  # auc为Roc曲线下的面积
    return AUC, accuracy_score(y_test, predictions), precision_score(y_test, predictions), recall_score(y_test,predictions), f1_score(
        y_test, predictions),fpr,tpr

def nav_sklearn(data):
    AUC = []
    ACC = []
    PRE = []
    REC = []
    F1 = []
    fpr = []
    tpr = []
    num = [45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
    for i in range(len(num)):
        AUC_temp, ACC_temp, PRE_temp, REC_temp, F1_temp, fpr_temp, tpr_temp = nav_sklearn_son(data, num[i])
        AUC.append(AUC_temp)
        ACC.append(ACC_temp)
        PRE.append(PRE_temp)
        REC.append(REC_temp)
        F1.append(F1_temp)
        fpr.append(fpr_temp)
        tpr.append(tpr_temp)
    temp = np.argmax(ACC)
    print('朴素贝叶斯 ACC = ', ACC[temp], 'AUC = ', AUC[temp], 'PRE = ', PRE[temp], 'REC = ', REC[temp], 'f1 = ', F1[temp])
    print(ACC)
    return fpr[temp], tpr[temp], AUC[temp]
def nav_sklearn_son(data,num):
    columns_name = list(data.columns)
    columns_name.remove('safe_or_bad')
    columns_name.remove('app_name')
    # 随机采样构建训练测试集合
    X_train, X_test, y_train, y_test = train_test_split(data[columns_name], data['safe_or_bad'], test_size=num*0.01)
    # 标准化数据
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)
    # 初始化
    bayes = naive_bayes.GaussianNB()
    bayes.fit(X_train, y_train)
    predictions =bayes.predict(X_test)
    y_probs = bayes.predict_proba(X_test)[:, 1]  # 模型的预测得分
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    AUC = auc(fpr, tpr)  # auc为Roc曲线下的面积
    return AUC, accuracy_score(y_test, predictions), precision_score(y_test, predictions), recall_score(y_test, predictions), f1_score(
        y_test, predictions),fpr,tpr

def MLP_sklearn(data):
    AUC = []
    ACC = []
    PRE = []
    REC = []
    F1 = []
    fpr = []
    tpr = []
    num = [45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
    for i in range(len(num)):
        AUC_temp, ACC_temp, PRE_temp, REC_temp, F1_temp, fpr_temp, tpr_temp = MLP_sklearn_son(data, num[i])
        AUC.append(AUC_temp)
        ACC.append(ACC_temp)
        PRE.append(PRE_temp)
        REC.append(REC_temp)
        F1.append(F1_temp)
        fpr.append(fpr_temp)
        tpr.append(tpr_temp)

    temp = np.argmax(ACC)
    print('神经网络 ACC = ', ACC[temp], 'AUC = ', AUC[temp], 'PRE = ', PRE[temp], 'REC = ', REC[temp], 'f1 = ', F1[temp])
    print(ACC)
    return fpr[temp], tpr[temp], AUC[temp]
def MLP_sklearn_son(data,num):
    columns_name = list(data.columns)
    columns_name.remove('safe_or_bad')
    columns_name.remove('app_name')
    # 随机采样构建训练测试集合
    X_train, X_test, y_train, y_test = train_test_split(data[columns_name], data['safe_or_bad'], test_size=num*0.01)
    # 标准化数据
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)
    # 初始化
    mlp = MLPClassifier()
    mlp.fit(X_train, y_train)
    predictions = mlp.predict(X_test)
    y_probs = mlp.predict_proba(X_test)[:, 1]  # 模型的预测得分
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    AUC = auc(fpr, tpr)  # auc为Roc曲线下的面积
    return AUC, accuracy_score(y_test, predictions), precision_score(y_test, predictions), recall_score(y_test, predictions), f1_score(
        y_test, predictions),fpr,tpr

def XGBoost_sklearn(data):
    AUC = []
    ACC = []
    PRE = []
    REC = []
    F1 = []
    fpr = []
    tpr = []
    num = [45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
    for i in range(len(num)):
        AUC_temp, ACC_temp, PRE_temp, REC_temp, F1_temp, fpr_temp, tpr_temp= XGBoost_sklearn_son(data, num[i])
        AUC.append(AUC_temp)
        ACC.append(ACC_temp)
        PRE.append(PRE_temp)
        REC.append(REC_temp)
        F1.append(F1_temp)
        fpr.append(fpr_temp)
        tpr.append(tpr_temp)

    temp = np.argmax(ACC)
    print('集成学习xgboost ACC = ', ACC[temp], 'AUC = ', AUC[temp], 'PRE = ', PRE[temp], 'REC = ', REC[temp], 'f1 = ', F1[temp])
    print(ACC)
    return fpr[temp], tpr[temp], AUC[temp]
def XGBoost_sklearn_son(data,num):
    columns_name = list(data.columns)
    columns_name.remove('safe_or_bad')
    columns_name.remove('app_name')
    # 随机采样构建训练测试集合
    X_train, X_test, y_train, y_test = train_test_split(data[columns_name], data['safe_or_bad'], test_size=num * 0.01)
    #xgm = xgb.XGBClassifier(max_depth=3)
    xgm = xgb.XGBClassifier()
    #eval_set = [(X_test, y_test)]
    xgm.fit(X_train, y_train)
    y_pred = xgm.predict(X_test)

    predictions = [round(value) for value in y_pred]
    y_probs = xgm.predict_proba(X_test)[:, 1]  # 模型的预测得分
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    AUC= auc(fpr, tpr)  # auc为Roc曲线下的面积
    return AUC, accuracy_score(y_test, predictions), precision_score(y_test, predictions), recall_score(y_test,predictions), f1_score(
        y_test, predictions),fpr,tpr


def show_info_roc(str, color, fpr, tpr, AUC):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 开始画ROC曲线
    plt.plot(fpr, tpr, 'b', label=str +'(AUC = %0.6f)' % AUC,linestyle=":",color=color)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--', color='coral', linestyle=':', marker='|')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('False Positive Rate')  # 横坐标是fpr
    plt.ylabel('True Positive Rate')  # 纵坐标是tpr
    plt.title('ROC curve ')
    #plt.show()
    #plt.savefig(sys.path[0]+'\\roc\\'+str+'.png')
    plt.savefig(sys.path[0]+'\\source_code\\roc\\'+str+'.png', dpi=500)
    #plt.close()
def show_info_roc_son(str, color, fpr, tpr, AUC):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 开始画ROC曲线
    #plt.plot(fpr, tpr, 'b', label=str +'(AUC = %0.6f)' % AUC,linestyle=":",color=color)
    #plt.legend(loc='lower right')
    plt.xlim([0.0, 0.2])
    plt.ylim([0.8, 1.0])
    plt.xlabel('False Positive Rate')  # 横坐标是fpr
    plt.ylabel('True Positive Rate')  # 纵坐标是tpr
    plt.title('ROC curve ')
    #plt.show()
    #plt.savefig(sys.path[0]+'\\roc\\'+str+'.png')
    plt.savefig(sys.path[0]+'\\source_code\\roc_son\\'+str+'.png', dpi=500)
    #plt.close()


def show_ACC_picture(ACC,str):
    x = [45, 55, 65, 75, 85, 95]
    # print('逻辑回归准确率',b)
    plt.bar(x, ACC, 5, color='blue', align="center")
    for i in range(len(x)):
        plt.text(x[i], ACC[i], ACC[i])
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.xticks(x, ['45%', '55%', '65%', '75%', '85%', '95%'])
    plt.title(str+'不同测试集准确率分布情况')
    plt.xlabel('参数')
    plt.ylabel('准确率')
    plt.legend(str)
    plt.axis([40, 100, 0.8, 1.02])
    #plt.show()
    plt.savefig(picture_path+'\\' + str+'.png')
    plt.close()

if __name__ == '__main__':
    # 读取数据
    warnings.filterwarnings("ignore")
    df = pd.read_csv(sys.path[0]+'\\source_code\\table\\SDM_1_pro.csv')

    fpr, tpr, AUC = Decision_sklearn(df)
    show_info_roc('DT', 'red' ,fpr, tpr, AUC)
    show_info_roc_son('DT', 'red', fpr, tpr, AUC)
    
    fpr, tpr, AUC =nav_sklearn(df)
    show_info_roc('NB','plum', fpr, tpr, AUC)
    show_info_roc_son('NB', 'plum', fpr, tpr, AUC)

    fpr, tpr, AUC,ACC =Logistic_sklearn(df)
    show_info_roc('LR','turquoise', fpr, tpr, AUC)
    show_info_roc_son('LR', 'turquoise', fpr, tpr, AUC)
    #show_ACC_picture(ACC,'3.逻辑回归')

    fpr, tpr, AUC =svm_sklearn(df)
    show_info_roc('SVM','deeppink', fpr, tpr, AUC)
    show_info_roc_son('SVM', 'deeppink', fpr, tpr, AUC)

    fpr, tpr, AUC =XGBoost_sklearn(df)
    show_info_roc('XGBoost','green', fpr, tpr, AUC)
    show_info_roc_son('XGBoost', 'green', fpr, tpr, AUC)

    fpr, tpr, AUC =KN_sklearn(df)
    show_info_roc('K-NN','orangered', fpr, tpr, AUC)
    show_info_roc_son('K-NN', 'orangered', fpr, tpr, AUC)

    fpr, tpr, AUC =MLP_sklearn(df)
    show_info_roc('ANN','grey', fpr, tpr, AUC)
    show_info_roc_son('ANN', 'grey', fpr, tpr, AUC)
   

    fpr, tpr, AUC = Random_sklearn(df)
    show_info_roc('RF', 'purple', fpr, tpr, AUC)
    show_info_roc_son('RF', 'purple', fpr, tpr, AUC)

