import tensorflow as tf

Keras = tf.keras
Sequential = Keras.models.Sequential
Dense = Keras.layers.Dense
Dropout = Keras.layers.Dropout
Flatten = Keras.layers.Flatten
Conv2D = Keras.layers.Conv2D
MaxPooling2D = Keras.layers.MaxPooling2D
K = Keras.backend

import sys
import os
sys.path.append('/home/jjohn273/git/CMS-Classification')
from cms_modules.keras_callbacks import EpochTimerCallback

batch_size = 128
num_classes = 10
epochs = 10

img_rows, img_cols = 28, 28

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)


x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# ### Build CNN
kernel_size = (3, 3)

model = Sequential()
model.add(Conv2D(32, kernel_size,
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, kernel_size, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size, activation='relu'))
model.add(Conv2D(256, kernel_size, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
                                  
optimizer = tf.keras.optimizers.Adadelta()
loss = tf.keras.losses.sparse_categorical_crossentropy
epochTimer = EpochTimerCallback('./cpu_timings.csv')

model.compile(optimizer, loss, metrics=['accuracy'])

model.summary()

history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[epochTimer])
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

epochTimer.write_timings()
