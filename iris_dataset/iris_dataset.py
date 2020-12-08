# sklearn dataset : https://scikit-learn.org/stable/datasets/index.html

from sklearn.datasets import load_iris

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, fbeta_score

from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, StratifiedKFold

dataset = load_iris()
print(dataset.keys()) #dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])

print(dataset.feature_names)    #X컬럼이름
print(dataset.data[:5])         #X값

print(dataset.target_names)
print(dataset.target[:10])      #y값

# print(dataset.DESCR)

#sepal length (cm)  sepal width (cm)  ...  petal width (cm)

df = pd.DataFrame(data=dataset.data,
                  #columns=dataset.feature_names
                  columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
                  )
df["target"] = dataset.target
print(df.head())
print(df.info())
print(df.shape)

#------------ 결측X, 수치형 : 분석데이터
#분석 -------- 상관분석 : 피쳐관계
#---------------------: 일반피쳐:다중공선 타켓피쳐:상관도

# df.hist(figsize=(20,20))
# plt.show()
#
#
# sns.heatmap(df.corr(), cmap='Blues')
# plt.show()
#
X = df.iloc[: , :-1]
y = df.iloc[: , -1]
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=160)
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
pred = rf.predict(X_test)
print("전처리 전 우선 점수부터 확인------\n")
report = classification_report(y_test, pred)
print(report)

scaler = StandardScaler()
X_scaler = scaler.fit_transform(X)
X_train,X_test,y_train,y_test = train_test_split(X_scaler, y, test_size=0.2, random_state=160)
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
pred = rf.predict(X_test)
print("정규화 후 점수 확인------\n")
report = classification_report(y_test, pred)
print(report)

# KFold, StratifiedKFold
score_list = []
kfold = KFold(n_splits=3, random_state=11)
for k, (train_idx, test_idx) in enumerate(kfold.split(X_scaler)):
    X_train, X_test = X_scaler[train_idx], X_scaler[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(k, len(y_test), y_test[:10])

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)

    #분류의 모든 score는 average='binary'(default)
    #멀티클래스분류인 경우 : 'micro', 'macro', 'samples','weighted'

    f1 = f1_score(y_test, pred, average='macro')
    score_list.append(f1)
    print(k, "F1:", f1)
print("Kfold-cv3 F1:평균점수:", np.mean(score_list))



skfold = StratifiedKFold(n_splits=3, random_state=11)
for k, (train_idx, test_idx) in enumerate(skfold.split(X_scaler, y)):
    X_train, X_test = X_scaler[train_idx], X_scaler[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(k, len(y_test), y_test[:10])

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)

    #분류의 모든 score는 average='binary'(default)
    #멀티클래스분류인 경우 : 'micro', 'macro', 'samples','weighted'

    f1 = f1_score(y_test, pred, average='macro')
    score_list.append(f1)
    print(k, "F1:", f1)
print("Kfold-cv3 F1:평균점수:", np.mean(score_list))



scores5 = cross_val_score(model, X_sacler, y , cv=5, scoring='f1',verbose=0)


estimator, param_grid, *, scoring=None,
                 n_jobs=None, iid='deprecated', refit=True, cv=None,
                 verbose=0, pre_dispatch='2*n_jobs',
                 error_score=np.nan, return_train_score=False):
        super().__init__(
model = RandomForestClassifier()
#cross_val_score +  param_grid(튜닝) / refit=True(best모델반영)
GCV_model = GridSearchCV(model, refit=True, cv=5, scoring='f1_macro',verbose=0)
GCV_model.fit(X_train, y_train)
print(GCV_model.best_score_)
print(GCV_model.best_estimator_)
print(GCV_model.best_params_)


#pred = rf.predict(X_test)


from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
##정규화 사용 이후 사용 : 최대분산(수치,스케일에 민감)
scaler = StandardScaler()
pca = PCA(n_components=2)
pipeline = make_pipeline(scaler, pca)
pca_res = pipeline.fit_transform(X)
print(pca.explained_variance_ratio_)
# pca = PCA()
# res = pca.fit_transform(X_scaler)
print(pca_res.shape)

pca_df = pd.DataFrame(data=pca_res
                      , columns=["pc1", "pc2"])
pca_df["target"] = y
pca_df.head()
