#Module Name : adv_house_price.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from scipy.stats import skew #왜도
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings(action="ignore")

df = pd.read_csv("train.csv")
print(df.shape)

# ---------- dataframe 타입별(카테고리형/숫치형) 피쳐 분류
def type_check(df):
    object_feature = df.dtypes[df.dtypes == 'object'].index
    numeric_feature = df.dtypes[df.dtypes != 'object'].index
    # numeric_column = df.columns - object_column
    print(df.dtypes.value_counts())
    print("object type:",object_feature)  #숫치형->범주화/인코딩
    print("numeric type:",numeric_feature) #스케일,범주화
    return object_feature, numeric_feature

from sklearn.linear_model import Ridge, Lasso, LinearRegression

#대량의 피쳐 분석의 경우 : 결측치 피쳐 삭제부터
# ---------- dataframe 결측치 확인/제거
def null_feature_check(df, drop_cnt=None):
    isnull_feature = df.isnull().sum()[df.isnull().sum()>0].index
    null_df = pd.DataFrame()
    null_df["null_cnt"]  = df[isnull_feature].isnull().sum()
    null_df["null_rate"]  = df[isnull_feature].isnull().sum() / df.shape[0] * 100
    print(null_df)

    # FireplaceQu     690 47.260274  --- 모델 평가 후 재검토 필요
    if drop_cnt != None :
        drop_isnull_feature = df.isnull().sum()[df.isnull().sum() > drop_cnt].index
        #['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature']
        print("drop_feature:",drop_isnull_feature)

        # df.drop(drop_isnull_feature, axis=1, inplace=True)
        df = df.drop(drop_isnull_feature, axis=1)
        print(df.shape)  #(1460, 76)
    return df

# def objct_type_fillna(df, object_feature):
#     #불가능
#     df.fillna('None', inplace=True)
#     #불가능
#     df.fillna(np.nan).replace(np.nan, 'None', inplace=True)
#     #가능
#     for col in object_feature:
#         df[col].fillna('None', inplace=True)
#
# def numeric_type_fillna(df, numeric_feature):
#     #가능
#     df.fillna(df.mean(), inplace=True)
#     #가능
#     for col in numeric_feature:
#         df[col].fillna(df[col].mean(), inplace=True)

df = null_feature_check(df)

df.t
df = null_feature_check(df, 600)
object_feature, numeric_feature = type_check(df)
# objct_type_fillna(df, object_feature)
# numeric_type_fillna(df, numeric_feature)
df.fillna(df.mean(), inplace=True)
df.replace(np.nan, 'None', inplace=True)

print("=============")
df = null_feature_check(df)


y = df["SalePrice"]
X = df.drop(["SalePrice"], axis=1)
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, random_state=11, test_size=0.2)

#============== 정규분포 확인
sk_df = pd.DataFrame()

#============== skew() kurt() 왜도/첨도 수치확인
# sns.distplot(df[col])
#왜도 : 대칭=0(정규분포) <0 :오른쪽편중  >0:왼쪽편중
sk_df["skew"] = df[numeric_feature].skew()
#첨도 : 뾰족함(중앙 편중)
sk_df["kurt"] = df[numeric_feature].kurt()
print(sk_df.head(40).sort_values("skew", ascending=False))

df.hist(bins=10, figsize=(16,16), grid=False);
plt.show()


# MinMaxScaler: 0 ~ 1
# StandardScaler: 평균0 분산1(정규분포)
# np.log()
#  : max을 0으로두고 다른값을 뺀 그 차이값을 이용
# np.log1p()
# : log를 취한값이 너무 작으면 언더플로우가 나기 때문에 1을 더해서 사용

sns.distplot(df['SalePrice'])
plt.show()

df['SalePrice2'] = np.log1p(df['SalePrice'])
sns.distplot(df['SalePrice2'])
plt.show()

# select distinct deptno from emp;
#수치형 : 0/1/2 코드성 수치
oh_numeric_feature = []
for col in numeric_feature:
    if df[col].nunique() < 10:
        print(col, df[col].unique())
        oh_numeric_feature.append(col)
print("인코딩 적용 전 :",df.shape)
oh_df = pd.get_dummies(data=df, columns=oh_numeric_feature)
print("인코딩 적용 후 :",oh_df.shape)
# print(df.head())


skew_feature = sk_df.head(40).sort_values("skew", ascending=False)[:15].index
skew_feature = skew_feature # - oh_numeric_feature
for col in skew_feature:
    df[col] = np.log1p(df[col])
df.hist(bins=10, figsize=(16,16), grid=False);
plt.show()


#========== IRQ를 사용한 Oulier 탐지
from collections import Counter
def detect_outliers(df, n, features):
    outlier_idx_list = []
    for col in features:
        Q1 = np.percentile(df[col], 25)
        Q3 = np.percentile(df[col], 75)
        IQR = Q3 - Q1
        IQR15 = 1.5 * IQR
        outlier_list_col = df[(df[col] < Q1 - IQR15) | (df[col] > Q3 + IQR15)].index
        outlier_idx_list.extend(outlier_list_col)
    outlier_idx_list = Counter(outlier_idx_list)
    multiple_outliers = list(k for k, v in outlier_idx_list.items() if v > n)
    return multiple_outliers
outlier_drop_feature = detect_outliers(df, 2, numeric_feature)
print(outlier_drop_feature)

print("Outlier 처리전:", df.shape)
df = df.drop(outlier_drop_feature, axis = 0).reset_index(drop=True)
print("Outlier 처리후:", df.shape)

#============== matplot.plt(box)
# plt.figure(figsize=(16,16))
# df.plot(kind='box', subplots=True, layout=(4,10), sharex=False, sharey=False)
# plt.show()



#============== sns.boxplot()
fig, axs = plt.subplots(figsize=(16, 8), ncols=4, nrows=10)
for i, feature in enumerate(numeric_feature):
    row = int(i / 4)
    col = i % 4
    # seaborn의 regplot 이용하여 박스플롯
    sns.boxplot(x=feature, y='SalePrice', data=df, ax=axs[row][col])
plt.show()

plt.figure(figsize=(16,16))
sns.heatmap(df[numeric_feature].corr(), annot=True)
plt.show()

saleprice_corr_feature \
    = df[numeric_feature].corr().nlargest(15, "SalePrice").index
print(saleprice_corr_feature)
sns.heatmap(df[saleprice_corr_feature].corr(), annot=True, cmap="Blues")
plt.show()

lr = LinearRegression()
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=1.0)
models = [lr, ridge, lasso]
for model in models:
    model.fit()
    model.predict()
    mse = mean_squared_error()
    print(model.__class__.__name__, mse, np.sqrt(mse))

for model in models:
    neg_mse5 = -1 * cross_val_score(model, X=X_oh, y=y_log
                    , scoring="neg_mean_squared_error"
                    , verbose=0, cv=5)
    print(model.__class__.__name__,
          np.mean(neg_mse5), np.sqrt(np.mean(neg_mse5)))

cf_df = pd.DataFrame()
for model in models:
    cf_df["coef"] = model.coef_
    cf_df["col"] = X_train.columns
    cf_df.sort_values(ascending=False)[:15]
    print(cf_df.head(15))


myprm = {'alpha':[0.05,0.1,0.5,1.0,3.0,10.0, 20.0]}
best_model = GridSearchCV(ridge, param_grid=myprm,
             scoring="neg_mean_squared_error",verbose=0,
             refit=True, cv=5)
best_model.fit(X_train7, y_train7)
print(best_model.best_score_)
print(best_model.best_params_)

best_pred = best_model.predict(X_test3)
mse = mean_squared_error(y_test3, best_pred)
print(mse, np.sqrt.sqre(mse))

np.expm1()