from __future__ import print_function
import pandas as pd
import numpy as np 
import keras
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
config = tf.ConfigProto(intra_op_parallelism_threads=4,\
        inter_op_parallelism_threads=4, allow_soft_placement=True,\
        device_count = {'CPU' : 1, 'GPU' : 1})
session = tf.Session(config=config)
K.set_session(session)
batch_size = 128
num_classes = 10
epochs = 30

img_rows, img_cols = 28, 28
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape[0], 'train samples')

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
    
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(64, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(128, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

newxs=np.array(pd.read_csv('train.csv').drop(['label'],1))
newys= np.array(pd.read_csv('train.csv')['label'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          )

model.fit(x_test, y_test,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          )
newys=keras.utils.to_categorical(newys, num_classes)
if K.image_data_format() == 'channels_first':
    newxs=newxs.reshape(newxs.shape[0],1, img_rows, img_cols)
    newxs=newxs.astype('float32')
    newxs/= 255
    model.fit(newxs,newys,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          )
    predictions=pd.DataFrame(np.argmax(model.predict(np.array(pd.read_csv('test.csv')).reshape(np.array(pd.read_csv('test.csv')).shape[0], 1, img_rows, img_cols)),1))
else:
    newxs=newxs.reshape(newxs.shape[0],img_rows, img_cols,1)
    newxs=newxs.astype('float32')
    newxs /= 255
    model.fit(newxs,newys,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          )
xpred=np.array(pd.read_csv('test.csv'))
xpred=xpred.reshape(xpred.shape[0],img_rows, img_cols,1)
xpred=xpred.astype('float32')
xpred /= 255
predictions=pd.DataFrame(np.argmax(model.predict(xpred),1))

predictions.index+=1
predictions.to_csv('pred.csv')