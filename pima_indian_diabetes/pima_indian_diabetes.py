# Module Name: pima_indian_diabetes.py

import pandas as pd
import numpy as np

#EDA-chart / 수치화,결측처리-> corr()
import matplotlib.pyplot as plt
import seaborn as sns

#전처리 : outlier, scale(정규/표준), encoding
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, Binarizer #pd.get_dummpy()

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import Xg.. li...
# from sklearn.metrics import mean_squared_error ... mse rmse mas rmsle
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV

import warnings
warnings.filterwarnings(action='ignore')

df = pd.read_csv("diabetes.csv")
y = df.iloc[:,-1]
X = df.iloc[:,:-1]
print(X.shape, y.shape)
print(df.info())

#============= 결측처리 / 숫자형 --> EDA
df.hist(figsize=(6,6))
#plt.show()

#============= EDA(상관분석)
sns.heatmap(df.corr(), cmap='Blues', annot=True, fmt='0.2f')
#plt.show()


#============= 전처리 : oulier 일환 = 0값 찾기
features = df.columns
print(type(features))
for feature in features:
    zero_cnt = df[df[feature]==0][feature].count()
    print(feature, zero_cnt, zero_cnt/df.shape[0]*100)

#============= 정규화 : 평균0 분산1로 스케일 조정
scaler = StandardScaler()
X_scaler = scaler.fit_transform(X)
X_scaler_df = pd.DataFrame(X_scaler, columns=X.columns)
X_scaler_df.hist()
#plt.show()

fig, axes = plt.subplots(figsize=(16,4), ncols=2)
plt.xticks(range(0, 30000, 1000), rotation=60)

bins = np.arange(df['hour'].min(),df['hour'].max()+2)

sns.distplot(df['Amount'] , ax=axes[0], bins=bins)
axes[0].set(ylabel='Count',title="연도별 대여량")
sns.distplot(df['Time'] , ax=axes[1])
plt.show()



X_train,X_test,y_train,y_test = train_test_split(X_scaler_df, y, test_size=0.2, random_state=160)

dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
pred = rf.predict(X_test)
print(pred[:10])  #당뇨인지 아닌지 내가 예측한거

def myscore(y_test, pred):
    accuracy = accuracy_score(y_test, pred)
    recall = recall_score(y_test, pred)
    precision = precision_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    confusion = confusion_matrix(y_test, pred)
    print("정확도:{:.4f},재현율:{:.4f},정밀도:{:.4f},f1:{:.4f}".format(accuracy,recall,precision,f1))
    print("confusion:", confusion)
myscore(y_test, pred)

X_train,X_test,y_train,y_test = train_test_split(X_scaler_df, y, test_size=0.2, random_state=160)
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
pred = rf.predict(X_test)
proba = rf.predict_proba(X_test)[:,1] #Positive
print(pred[:10])  #당뇨인지 아닌지 내가 예측한거
print(proba[:10])

pscore,rscore,th = precision_recall_curve(y_test, proba)
print(len(pscore), len(rscore), len(th)) #58 58 57

plt.plot(th, pscore[:len(th)], label="precision")
plt.plot(th, rscore[:len(th)], label="recall")
plt.xlabel("thredholds")
plt.ylabel("precision recall value")
plt.legend()
plt.grid()
plt.show()

fpr,tpr,th = roc_curv(y_test, proba) #th: max(score_len) + 1
plt.plot(fpr, tpr, label="roc")
plt.plot([0,1], [0,1], label="th:0.5")
plt.xlabel("FPR(1-sen.)")
plt.ylabel("TPR(recall)")
plt.grid()
plt.show()

auc = roc_auc_score(y_test, rf.predict_proba(X_test))
print(auc)


print("정규화 후 점수 확인------\n")
myscore(y_test, pred)




thresholds = [0.3,0.4,0.5,0.55, 0.6]
for th in thresholds:
    binarizer = Binarizer(threshold=th)
    pred = binarizer.fit_transform(proba.reshape(-1,1))
    print(th, pred, proba)#P.s

## dt .... voting bagging boosting
#X_train,X_test,y_train,y_test
def fit_predict(model):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    myscore(y_test, pred, proba)

from sklearn.linear_model import LogisticRegression
penalty = ['l1', 'l2']
C= [0.5, 1.0, 3.0]
for p in penalty:
    for c in C:
        model = LogisticRegression(random_state=11,penalty=p, C=c)
        fit_predict(model)


from sklearn.ensemble import GradientBoostingClassifier
learning_rate_value = [0.01,0.1,1.0]
n_estimators_value = [50,100,300]
for lr in learning_rate_value:
    for est in n_estimators_value:
        print("lr, 트리갯수", lr, est)
        model = GradientBoostingClassifier(random_state=11,learning_rate=lr, n_estimators=est)
        fit_predict(model)

from xgboost import XGBClassifier
n_estimators_value = [100,300,500]
max_depth = [3,5,7]
#과적합 :: min_child_weight(높게) / gamma(노드분기:크게)
#         eta(learnrate:작게)
for est in n_estimators_value:
    for depth in max_depth:
        print("lr, 트리갯수", lr, est)
        model = XGBClassifier(objective="binary:logistic",
                            early_stopping_rounds=10,
                            eval_metric="auc",
                            n_estimators=est,
                            max_depth = depth
        )
        fit_predict(model)

for lr in learning_rate_value:
    for est in n_estimators_value:
        print("lr, 트리갯수", lr, est)
        model = GradientBoostingClassifier(random_state=11,learning_rate=lr, n_estimators=est)
        fit_predict(model)
from sklearn.ensemble import VotingClassifier
ensemble_model = VotingClassifier(model_list)



model = XGBClassifier(objective="binary:logistic",
                            #early_stopping_rounds=10,
                            eval_metric="auc",
                            n_estimators=est,
                            max_depth = depth
        )


hyper_param = { "n_estimators":[50,100,200,300,400],
                "max_depth":[3,4,5,6,7,8]}
model = XGBClassifier(objective="binary:logistic",
                            #eval_metric="auc",
                            #n_estimators=est,
                            #max_depth = depth
        )
grid_model = GridSearchCV(estimator=model,
                          param_grid=hyper_param,
             scoring='roc_auc',  refit=True, cv=5, verbose=0)
# fit_predict(grid_model)
gird_model.pretict(X_train, y_train)
print(grid_model.best_params_)
print(grid_model.best_score_)

df.pivot_table(values='Amount',index='hour',columns='Class',aggfunc='count').sort_values(ascending=False)
sns.heatmap(df.corr(), cmap='Blues', annot=True, fmt='0.2f')
plt.figure(figsize=(14,14))
best_model = GridSearchCV(estimator=model,param_grid=mydic,
             scoring='roc_auc',  refit=True, cv=5, verbose=0)

