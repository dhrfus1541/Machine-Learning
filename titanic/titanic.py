# encoding
# Module Name : titanic.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plgt
import seaborn as sns
import warnings
warnings.filterwarnings(action="ignore")

# from sklearn.datasets import ____
# from sklearn.metrics import ____
# from sklearn. import ____

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# shape describe() info() head()
train_df.shape  #750 * 13
train_df.info()
train_df.head()  #head(5)
train_df.describe()


# train_df["Age"] = train_df["Age"].fillna(train_df["Age"].mean())
train_df["Age"].fillna(train_df["Age"].mean(), inplace=True)
train_df["Age"].fillna(train_df["Age"].mean(), inplace=True)


train_df["Embarked"].value_counts()
train_df["PassengerId"].drop(axis=1, inplace=True)

train_df["PC"] = train_df["SibSp"] + train_df["Parch"]

train_df[["SibSp","Parch"]].drop(axis=1, inplace=True)

select pclass,sum(serv) from df
group by pclss
order by serv desc

train_df[["Pclass","Servived"]].groupby(["Pclass"]).sum().sort_values(by="Servived", ascending=False)
