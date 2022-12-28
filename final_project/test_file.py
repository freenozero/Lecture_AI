import pathlib
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, VGG16

#사이즈 지정
img_size = 224

test_path = pathlib.Path('C:/Users/freet/Desktop/3학년2학기/인공지능/과제/꽃 분류/test_data')
test_img = list(test_path.glob('*/*.jpg'))

x_test, y_test, y_class = [], [], []
y = {'cosmos': 0, 'daisy': 1, 'hollyhock': 2, 'marigold':3 } #class 분류


#test set    
for path in test_img:
    img = image.load_img(path, target_size=(img_size, img_size))
    img = image.img_to_array(img)
    x_test.append(preprocess_input(img)) #VGG 모델을 위한 전처리
    #class 분류 숫자 저장
    sp = str(path).split(sep='\\') 
    y_class.append(sp[9]) #이후에 어떤 클래스가 잘 예측했는지에 사용
    y_test.append(y.get(sp[9]))


y_test = tf.keras.utils.to_categorical(y_test)


x_test = np.array(x_test)
y_test = np.array(y_test)


# for i in x_test:
#     print(i)
# for i in y_test:
#     print(i)

#모델 전체 로드
model = tf.keras.models.load_model("C:/Users/freet/Desktop/3학년2학기/인공지능/과제/꽃 분류/ckpt/two/RES/two.h5")

#weights
#latest = tf.train.load_checkpoint("C:/Users/freet/Desktop/3학년2학기/인공지능/과제/꽃 분류/ckpt/zero/")
model.load_weights(r"C:\Users\freet\Desktop\3학년2학기\인공지능\과제\꽃 분류\ckpt\two\two-0007.ckpt")

#모델 평가 예측 
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

y_pred = model.predict(x_test, verbose=0)
y_pred = np.around(y_pred)
y_act = np.around(y_test)


pred_class = {'cosmos': 0, 'daisy': 0, 'hollyhock': 0, 'marigold':0 } #class 분류
for i in range(len(y_pred)):
    print(y_class[i], ": ", end='')
    # print(y_pred[i])
    # print(y_act[i])
    if(np.all(y_pred[i] == y_act[i])):
        print("true")
    else:
        print("false")
