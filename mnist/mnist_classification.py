# from sklearn.datasets import fetch_openml
# mnist = fetch_openml('mnist_784')
# mnist.data.shape, mnist.target.shape
# # (70000, 784)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.datasets import load_digits  #MNIST 손글씨

dataset = load_digits()
print(dataset.keys())

X = dataset.data    #이미지 0~255 픽셀값
y = dataset.target  #숫자값 0~9
print(X.shpae, y.shape)
print(X[:1], y[0])

# np.array 64  --> 8*8
for i in range(0,10):
    plt.imshow(  np.reshape(X[i],(8,8))  )
    plt.xlabel(y[i])
    plt.show()

X_train,X_test, y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=111)
print(X_train.shape,X_test.shape, y_train.shape,y_test.shape )

model = RandomForestClassifier()
model.fit(X_train, y_train)
pred = model.fit(X_test)
report = classification_report(y_test, pred)
print(report)

scaler = StandardScaler()
X_train_sacle = scaler.fit_transform(X_train)
X_test_sacle = scaler.fit_transform(X_test)
model.fit(X_train_sacle, y_train)
pred = model.fit(X_test)
report = classification_report(y_test, pred)
print(report)