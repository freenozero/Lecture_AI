#45_2, 45_3, 45_04, 49_05, 50_01 참고.
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import random
import os
from tensorflow.keras.layers import Input, Dense, Flatten, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications.vgg16 import preprocess_input, VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
#This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations: AVX AVX2 To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


#사이즈 지정
img_size = 224

#train, test path
#C:/Users/freet/Desktop/3학년2학기/인공지능/과제/꽃 분류/train_data
#C:\Users\freet\Desktop\3학년2학기\인공지능\과제\꽃 분류\train_data_rotate
train_path = pathlib.Path(r'C:\Users\freet\Desktop\3학년2학기\인공지능\과제\꽃 분류\train_data_rotate')
val_path = pathlib.Path(r'C:\Users\freet\Desktop\3학년2학기\인공지능\과제\꽃 분류\val_data_rotate')

#이미지 list에 저장
train_img = list(train_path.glob('*/*.jpg'))
val_img = list(val_path.glob('*/*.jpg'))
#validation set이 생겨서 shuffle 불필요
# random.shuffle(train_img)

#영상 읽어오기
x_train, y_train = [], []
x_val, y_val = [], []
y = {'cosmos': 0, 'daisy': 1, 'hollyhock': 2, 'marigold':3 } #class 분류
#train set
for path in train_img:
    #print(path)
    img = image.load_img(path, target_size=(img_size, img_size))
    img = image.img_to_array(img)
    
    x_train.append(preprocess_input(img)) #VGG 모델을 위한 전처리
    #class 분류 숫자 저장
    sp = str(path).split(sep='\\') 
    y_train.append(y.get(sp[9]))
#val set
for path in val_img:
    #print(path)
    img = image.load_img(path, target_size=(img_size, img_size))
    img = image.img_to_array(img)
    
    x_val.append(preprocess_input(img)) #VGG 모델을 위한 전처리
    #class 분류 숫자 저장
    sp = str(path).split(sep='\\') 
    y_val.append(y.get(sp[9]))

#원-핫 인코딩
y_train = tf.keras.utils.to_categorical(y_train) 
y_val = tf.keras.utils.to_categorical(y_val)
# 잘 저장 되었는지 확인
# for i in X:
#     print(i)
# for i in x_train:
#     print(i)
# for i in y_train:
#     print(i)

#list를 np.array로 변경
x_train = np.array(x_train)
y_train = np.array(y_train)
x_val = np.array(x_val)
y_val = np.array(y_val)

#input
inputs = Input(shape=(img_size, img_size, 3))
#model 불러오기
model = VGG16(weights="imagenet", classes=4, input_shape=(img_size, img_size, 3), include_top=False, input_tensor=inputs)
model.trainable = False

#output
x = model.output
x = Flatten()(x)
# x = Dense(1024, activation = 'relu')(x)
# outs = Dense(4, activation = 'softmax')(x)
x = Dense(4096, activation = 'relu')(x)
x = Dense(4096, activation = 'relu')(x)
outs = Dense(4, activation = 'softmax')(x)
model = tf.keras.Model(inputs, outs)
model.summary()


opt = tf.keras.optimizers.Adam(learning_rate=0.001) #0.001
model.compile(optimizer = opt, loss="categorical_crossentropy", metrics=['accuracy'])  # categorical_crossentropy, mse

#모델 전체 저장
if not os.path.exists("C:/Users/freet/Desktop/3학년2학기/인공지능/과제/꽃 분류/ckpt/two/RES"):
    os.mkdir("C:/Users/freet/Desktop/3학년2학기/인공지능/과제/꽃 분류/ckpt/two/RES")
model.save("C:/Users/freet/Desktop/3학년2학기/인공지능/과제/꽃 분류/ckpt/two/RES/two.h5")
#체크포인트, 가중치만 저장, 1 epoch마다 저장
filepath = "C:/Users/freet/Desktop/3학년2학기/인공지능/과제/꽃 분류/ckpt/two/two-{epoch:04d}.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath, verbose=0, save_best_only=True, save_weights_only=True, save_freq='epoch', monitor='val_accuracy')

#batch_size를 64로 하니까 할당이 메모리 10%를 초과한다고 떠서 32로 변경
ret = model.fit(x_train, y_train, epochs=15, batch_size=32, validation_data=(x_val, y_val), verbose=2, callbacks=[cp_callback])


fig, ax = plt.subplots(1, 2, figsize=(10, 6))
ax[0].plot(ret.history['loss'], "b-", label="train loss")
ax[0].plot(ret.history['val_loss'], "r-", label="var loss")
ax[0].set_title("loss")
ax[0].set_xlabel("epochs") 
ax[0].set_ylabel("loss")

ax[1].plot(ret.history['accuracy'], "b-",
           label="train accuracy")
ax[1].plot(ret.history['val_accuracy'], "r-",
           label="val accuracy")
ax[1].set_title("accuracy")
ax[1].set_xlabel("epochs")
ax[1].set_ylabel("accuracy")
plt.legend(loc="best") 
fig.tight_layout()
plt.show()