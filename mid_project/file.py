import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def load_ttt(shuffle=True):
    game_board = {'x': 0, 'o': 1, 'b': 2}

    game_win = {'false': 0, 'true': 1}

    data = np.loadtxt("./tic-tac-toe.csv", skiprows=1, delimiter=',',
                      converters={0: lambda board: game_board[board.decode()],
                                  1: lambda board: game_board[board.decode()],
                                  2: lambda board: game_board[board.decode()],
                                  3: lambda board: game_board[board.decode()],
                                  4: lambda board: game_board[board.decode()],
                                  5: lambda board: game_board[board.decode()],
                                  6: lambda board: game_board[board.decode()],
                                  7: lambda board: game_board[board.decode()],
                                  8: lambda board: game_board[board.decode()],
                                  9: lambda winner: game_win[winner.decode()]})

    # print(data[0], data[637])  # 치환 됐는지 확인

    if shuffle:
        np.random.shuffle(data)
    return data


ttt_data = load_ttt()
X = ttt_data[:, :-1]  # class 제외
y_true = ttt_data[:, -1]  # class만
y_true = tf.keras.utils.to_categorical(y_true)  # 원-핫으로 변경
# print(y_true[0])
# print("X.shape:", X.shape)  # 958x9 데이터
# print("y_true.shape:", y_true.shape)  # 958x2 데이터

# train, test = 8:2 or 9:1
X_train, X_test, y_train, y_test = train_test_split(
    X, y_true, test_size=0.2, stratify=y_true, random_state=1)

# print("X_train.shape", X_train.shape)
# print("y_train.shape", y_train.shape)
# print("X_test.shape", X_test.shape)
# print("y_test.shape", y_test.shape)


# 2층 신경망
# n = 2
n = 10
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=n, input_dim=9, activation="sigmoid"))
model.add(tf.keras.layers.Dense(units=2, activation='softmax'))
model.summary()

# opt = tf.keras.optimizers.RMSprop(learning_rate=0.01)
opt = tf.keras.optimizers.Adam(0.001)

model.compile(optimizer=opt, loss='binary_crossentropy',  # categorical_crossentropy
              metrics=['binary_accuracy'])  # accuracy

ret = model.fit(X_train, y_train, epochs=3000, verbose=2,
                validation_split=0.2, batch_size=64)  # validation_split=0.1

train_loss, train_acc = model.evaluate(X_train, y_train, verbose=2)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

y_pred = model.predict(X_train[:30], verbose=0)
y_pred = np.around(y_pred).astype(int).flatten()
y_act = np.around(y_train[:30]).astype(int).flatten()

print("pred:", y_pred)
print("act:    ", y_act)


fig, ax = plt.subplots(1, 2, figsize=(10, 6))
ax[0].plot(ret.history['loss'], "b-", label="train loss")
ax[0].plot(ret.history['val_loss'], "r-", label="var loss")
ax[0].set_title("loss")
ax[0].set_xlabel("epochs")
ax[0].set_ylabel("loss")

ax[1].plot(ret.history['binary_accuracy'], "b-",
           label="train accuracy")  # accuracy
ax[1].plot(ret.history['val_binary_accuracy'], "r-",
           label="val accuracy")  # val_accuracy
ax[1].set_title("accuracy")
ax[1].set_xlabel("epochs")
ax[1].set_ylabel("accuracy")
plt.legend(loc="best")
fig.tight_layout()
plt.show()
