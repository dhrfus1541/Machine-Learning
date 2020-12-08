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
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, fbeta_score

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, StratifiedKFold


dataset = load_iris()
df = pd.DataFrame(data=dataset.data,
                  #columns=dataset.feature_names
                  columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
                  )
# df["target"] = dataset.target

from sklearn.cluster import KMeans
model = KMeans(n_clusters=3, init='k-means++', max_iter=300, random_state=11)
model.fit(df[['sepal_length', 'sepal_width']])
cluster_pred = model.predict(df[['sepal_length', 'sepal_width']])

df["cluster_pred"] = cluster_pred
df["cluster_labels"] =model.labels_
print(df.head())

center_df = pd.DataFrame(model.cluster_centers_,columns=['sepal_length', 'sepal_width'] )
cx = center_df["sepal_length"]
cy = center_df["sepal_width"]
plt.scatter(cx, cy, c="r", marker="*")
color = ["b","y","g"]
plt.scatter(df["sepal_length"], df["sepal_width"], c=color[df["cluster_pred"]])
plt.show()



from sklearn.metrics import silhouette_score, silhouette_samples
scoef = silhouette_samples(df[['sepal_length', 'sepal_width']], df["cluster_pred"])
df["scoef"] = scoef
df.tail()


scaler = StandardScaler()
scaler.fit_transform(df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']])


scaler = StandardScaler()
scaler_data = scaler.fit_transform(df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']])
df_scaler = pd.DataFrame(scaler_data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
print(scaler_data)
df_scaler.head()


df_scaler["target"] = dataset.target
df_scaler.head()
ctab = pd.crosstab(df_scaler["target"],df_scaler["cluster_pred"])
print(ctab)

plt.plot(range(1,8), inertia_list)
plt.xlabel = "k-cluster"
plt.ylabel = "cluster inertia"
plt.xticks()
plt.show()


from scipy.cluster.hierarchy import linkage, dendrogram

plt.figure(figsize=(8,8))
matrix = linkage(df, method='single', metric='euclidean')
dendrogram(matrix)
plt.show()


from sklearn.cluster import DBSCAN
model = DBSCAN(eps=0.5, min_samples=5)

model.fit_predict()