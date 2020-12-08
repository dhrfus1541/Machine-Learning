#module name : emp_test.py
import numpy as np
# f = np.loadtxt("dd.txt")

# with f:
#
import pandas as pd
df = pd.read_csv("emp.csv", index_col="empno")  #, sep=",")
print(df.info())
print(df.head())
print(df.shape)
print(df.index.name)

#select ename from df
print(df["ename"])

#select ename,job from df
print(df[  ["ename", "job"]   ])

#select ename||job from df
print(df["ename"] + df["job"]) #SMITHSALES

#select sal*10 from df
# print(df["sal"]  ,   df["sal"]*10)
dfdf = pd.DataFrame()
dfdf["sal"] = df["sal"]
dfdf["sal10"] = df["sal"]*10

#----------- list --> Series
list = [1,2,3,4,5,6,7,8,9,10,11,12,13]
arr = np.array(list)
dfdf["list_seq"] = list #그냥 붙이면 된다
dfdf["arr_seq"] = arr #그냥 붙이면 된다

series = pd.Series(list)
dfdf["list_seq"] = (s for s in series)   #컬럼은 만들어지지만 데이터 값은 안들어옴

print(dfdf.head())
print(dfdf.shape)



# select sal,ename from df where sal > 1000
print(  df[["sal","ename"]][df["sal"]>3000]    )
print(  df[df["sal"]>3000][["sal","ename"]]   )
print(  df[df["sal"]>3000]["ename"]  )
print(  df[df["sal"]>3000].ename ) #,sal    )

# select sal,ename from df order by ename desc
print(  df[["sal","ename"]].sort_values(by="ename", ascending=False))

# select sal + bonus as CP from df
# df["Sipsp"] + df["Parch"]
df["CP"] = df["sal"] + df["bonus"]
print(df[["sal","bonus","CP"]].head())

# select bonus from emp where bonus is null
# print(df["ename"][df["bonus"] == np.nan])
print(df[df["bonus"].isnull() == True])
print(df[df["bonus"].isnull()])

#select deptno  from df
#select deptno, count(1) from df group by deptno
print(df["deptno"].value_counts())

# count min max sum avg
# count min mix sum mean
# select deptno, avg(sal) from df group by deptno
print(df.groupby(by="deptno")["sal"].mean())

#select ename, min(sal), max(sal) from df group by deptno
# print(df.groupby(by="deptno")["sal"].min()["sal"].max())
dict = {"sal":"max" , "sal":"min"}
print(df.groupby(by="deptno").agg(dict))
print(df.groupby(by="deptno")["sal"].agg([max, min]))

print(df.count())

#-----  결측 -----
print(df.isnull().sum() , df.isnull().sum()/df.shape[0] * 100 )
# deptno      0     0%
# CP          9     90%

df.loc[9998] = ["BBBB","MANAGER",7839,"",2850,0,30,0]
print(df.tail())

#-----  iloc[행번째,열번째]   loc[인덱스값,컬럼명]
#-----  iloc[s:e , s:e]
#-----  iloc[ :e , s:e]
#-----  iloc[ :  , -1]
#-----  iloc[ [3,4,5]  , 8]
#-----  iloc[ [3,4,5]  , [0,1,2]]
print(df.head())
print(df.iloc[0:2, 0:2])
print(df.iloc[0:2, [0,2] ])

# select ename, job from df where empno=7654 or empno=7698
print(df.loc[7499:7698, "ename":"mgrno"] )

# delete from df wehre empno=7499
df.drop(columns="hiredate", axis=1, inplace=True)   #세로(컬럼)삭제
# df.drop(["mgrno","job"], axis=1, inplace=True)    #세로(컬럼)삭제

df.drop(index=7499, axis=0, inplace=True)           #가로(레코드)삭제
print(df.head(15))


train_df["Name"] = train_df["Name"].str.extract("([A-Za-z]+)\.")