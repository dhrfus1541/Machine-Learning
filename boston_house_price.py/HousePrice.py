#Module Name : boston_house_price.py
import numpy as np
from sklearn.datasets import load_boston
import warnings
warnings.filterwarnings(action='ignore')

data = load_boston()

#['data', 'target', 'feature_names', 'DESCR', 'filename'])
print(data.keys())
print(data.feature_names)
# print(data.DESCR)

X = data.data
y = data.target   #11.9  $1,000
# print(y)

import pandas as pd
df = pd.DataFrame(data =data.data ,  columns=data.feature_names)
df["price"] = data.target
print(df.shape)  #506 *   12 + 1
#-------------------- EDA -------------

#-----------------학습 데이터 준비 7:3   8:2
from sklearn.model_selection import train_test_split
y = df["price"]
X = df.drop(["price"], axis=1)
학습문제X7,시험문제X3,학습단안y7,시험정답y3 = \
    train_test_split(X, y, random_state=11, test_size=0.2)
#train_test_split(data.data, data.target, random_state=11, test_size=0.2)


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
모델lr = LinearRegression()
모델rf = RandomForestRegressor()

모델rf.fit(학습문제X7, 학습단안y7)
예측답안y3 = 모델rf.predict(시험문제X3)

#---- 직선방정식(회귀계수==기울기==직선) 로부터 각 점들(실제y정답지) 간의 거리계산
#---- RSS : Y- Y^
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(시험정답y3,예측답안y3,squared=True)
rmse = mean_squared_error(시험정답y3,예측답안y3,squared=False)
print("7:3:" , mse, rmse)


from sklearn.model_selection import KFold
kf = KFold(n_splits=5,random_state=11)
mse_list = []
rmse_list= []

for k, (idx_train, idx_test) in enumerate(kf.split(X)):
    X_train, X_test = X.iloc[idx_train], X.iloc[idx_test]
    y_train, y_test = y.iloc[idx_train], y.iloc[idx_test]
    #---------- 이하코드 동일
    모델rf.fit(X_train, y_train)
    y_pred = 모델rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred, squared=True)
    #rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(k, mse, np.sqrt(mse))
    mse_list.append(mse)
    #rmse_list.append(rmse)

print("KFold-5회 MSE  평균:" , np.mean(mse_list))
print("KFold-5회 RMSE 평균:" , np.sqrt(np.mean(mse_list)))



from sklearn.model_selection import cross_val_score
score_arr5 = cross_val_score(모델rf, X, y, scoring="neg_mean_squared_error", cv=5)
print("CVS-5회 MSE:" , score_arr5)
print("CVS-5회 MSE 평균:" , np.mean(-score_arr5))
print("CVS-5회 RMSE 평균:" , np.sqrt(np.mean(-score_arr5)))
from sklearn.model_selection import GridSearchCV
hyper_param = {'max_depth':[2,4,6,8],
               'min_samples_leaf' : [1,2,3,4,5]}
gcv_model = GridSearchCV(모델rf, scoring="neg_mean_squared_error", cv=5,
                   param_grid=hyper_param,refit = True, verbose=0)
gcv_model.fit(X_train, y_train)
print("최고파라미터" , gcv_model.best_params_)
print("최고MSE" , gcv_model.best_score_)
print("최고모델" , gcv_model.best_estimator_)

y_pred = gcv_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred, squared=True)
print("GCV-튜닝 후 MSE 평균:" , np.mean(mse))
print("GCV-튜닝 후 RMSE 평균:" , np.sqrt(np.mean(mse)))


