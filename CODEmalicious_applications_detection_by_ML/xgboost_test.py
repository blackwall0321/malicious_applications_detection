
from sklearn.model_selection  import train_test_split
from sklearn import metrics
import xgboost as xgb
import pandas as pd
import sys
import warnings
from sklearn.metrics import precision_score, recall_score,accuracy_score,f1_score,roc_curve,auc
'''
  params = {
              'booster': 'gbtree', #linear booster
              'objective': 'binary:logistic',
              'eval_metric': 'auc',
              'max_depth': max_depth, #默认6  典型值：3-10  ************************
              'lambda': 10, #[默认1]
              'subsample': subsample, #[默认1] 典型值：0.5-1
              'colsample_bytree': colsample_bytree, #[默认1 典型值：0.5-1    ****************************
              'min_child_weight': 2,  #默认1
              'eta':eta,   #默认0.3 典型值为0.01-0.2            *************************
              'seed': 0,
              'nthread': 8,
              'silent': 1
                }
'''

f = open('E:\\result.txt','a')
def XGBoost_adjust_test(data,num):
    columns_name = list(data.columns)
    columns_name.remove('safe_or_bad')
    columns_name.remove('app_name')
    # 随机采样构建训练测试集合

    X_train, X_test, y_train, y_test = train_test_split(data[columns_name], data['safe_or_bad'],random_state=1,test_size=num * 0.01)
    for max_depth in range(3, 11, 1):
        for subsample in range(50, 105, 5):  # 100
            for colsample_bytree in range(5, 11, 1):  # 10
                for eta in range(1, 31, 1):  # 100
                    xgm = xgb.XGBClassifier(max_depth=max_depth,subsample=subsample*0.01,colsample_bytree=colsample_bytree*0.1,eta=eta*0.01)
                    xgm.fit(X_train, y_train)
                    y_pred = xgm.predict(X_test)
                    predictions = [round(value) for value in y_pred]
                    y_probs = xgm.predict_proba(X_test)[:, 1]  # 模型的预测得分
                    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
                    AUC = auc(fpr, tpr)  # auc为Roc曲线下的面积
                    #+' '+accuracy_score(y_test, predictions)+' '+precision_score(y_test, predictions)+' '+recall_score(y_test,predictions)+' '+f1_score(y_test, predictions)
                    print(str(max_depth)+' '+str(subsample*0.01)+' '+str(colsample_bytree*0.1)+' '+str(eta*0.01)+' '+str(accuracy_score(y_test, predictions)))
                    f.write(str(max_depth)+' '+str(subsample*0.01)+' '+str(colsample_bytree*0.1)+' '+str(eta*0.01)+' '+str(accuracy_score(y_test, predictions)))
def XGBoost_adjust_test1(data,num):
    columns_name = list(data.columns)
    columns_name.remove('safe_or_bad')
    columns_name.remove('app_name')
    # 随机采样构建训练测试集合

    X_train, X_test, y_train, y_test = train_test_split(data[columns_name], data['safe_or_bad'],random_state=1,test_size=num * 0.01)
    for max_depth in range(3, 11, 1):
        for subsample in range(50, 105, 5):  # 100
            for colsample_bytree in range(5, 11, 1):  # 10
                xgm = xgb.XGBClassifier(max_depth=max_depth,subsample=subsample*0.01,colsample_bytree=colsample_bytree*0.1)
                xgm.fit(X_train, y_train)
                y_pred = xgm.predict(X_test)
                predictions = [round(value) for value in y_pred]
                y_probs = xgm.predict_proba(X_test)[:, 1]  # 模型的预测得分
                fpr, tpr, thresholds = roc_curve(y_test, y_probs)
                AUC = auc(fpr, tpr)  # auc为Roc曲线下的面积
                print(str(max_depth)+' '+str(subsample*0.01)+' '+str(colsample_bytree*0.1)+' '+str(accuracy_score(y_test, predictions))+' ')
                f.write(str(max_depth)+' '+str(subsample*0.01)+' '+str(colsample_bytree*0.1)+' '+str(accuracy_score(y_test, predictions)))

warnings.filterwarnings("ignore")
df = pd.read_csv(sys.path[0] + '\\source_code\\table\\SDM_1_pro.csv')
XGBoost_adjust_test1(df,50)