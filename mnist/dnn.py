import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import classification_report  ##### NN에서 불필요
from sklearn.datasets import load_digits  #MNIST 손글씨

import warnings
warnings.filterwarnings(action='ignore')

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout

dataset = load_digits()
X = dataset.data    #이미지 0~255 픽셀값
                    #8*8 (64개 입력레이어 노드수)
y = dataset.target  #0~9 (10개 출력레이어 노드수)

# X_train,X_test, y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=111)
# print(X_train.shape,X_test.shape, y_train.shape,y_test.shape )


# ---------- sklearn용
scaler = StandardScaler()
X_scaler = scaler.fit_transform(X)
X_train,X_test,y_train,y_test = train_test_split(X_scaler, y, test_size=0.2, random_state=160)

# ---------- keras용
# (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# x_train = x_train.reshape(60000, 784).astype("float32") / 255
# x_test = x_test.reshape(10000, 784).astype("float32") / 255

model = Sequential()
model.add(Dense(64, input_dim=64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()  ##NN network 구성 확인
# from tensorflow.keras.utils import plot_model
# plot_model(model, "my_first_model.png"
#                        , show_shapes=True)


#binary_crossentropy
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])  #, 'f1'
#model = RandomForestClassfier()


#----------- 모델 최적화 설정 ######################
# from keras.callbacks import ModelCheckpoint,EarlyStopping
# MODEL_DIR = './model/'
# if not os.path.exists(MODEL_DIR):
#     os.mkdir(MODEL_DIR)
#
# mypath ="./model/{epoch:02d}-{val_loss:.4f}.hdf5"
# checkpointer = ModelCheckpoint(filepath=mypath, monitor='val_loss', verbose=1, save_best_only=True)
# early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

res = model.fit(X_train, y_train
                ,validation_data=(X_test,y_test)
                #, validation_split=0.2
                ,batch_size=100
                ,epochs=2,verbose=1
                # ,callbacks=[early_stopping_callback,checkpointer]
                )

#model.predict()



#--------------- 모델 평가
# val_loss val_accuracy
# loss accuracy
print(res.history['loss'])

score = model.evaluate(X_test, y_test)
print(model.metrics_names, score)


# -------------------- 그래프로 표현
val_loss = res.history['val_loss']
loss = res.history['loss']
x_len = np.arange(len(val_loss))
plt.plot(x_len, val_loss, marker='.', c="red", label='Testset_loss')
plt.plot(x_len, loss, marker='.', c="blue", label='Trainset_loss')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()


# ---------- keras용 모델 저장 ######################
# model.save("path_to_my_model")
# del model
# # Recreate the exact same model purely from the file:
# model = keras.models.load_model("path_to_my_model")

