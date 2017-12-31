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
#mini batch gradient descent ftw
batch_size = 128
#10 difference characters
num_classes = 10
#very short training time
epochs = 30

# input image dimensions
#28x28 pixel images. 
img_rows, img_cols = 28, 28

# the data downloaded, shuffled and split between train and test sets
#if only all datasets were this easy to import and format
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape[0], 'train samples')
#this assumes our data format
#For 3D data, "channels_last" assumes (conv_dim1, conv_dim2, conv_dim3, channels) while 
#"channels_first" assumes (channels, conv_dim1, conv_dim2, conv_dim3).
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
    
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    

#more reshaping
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#build our model
model = Sequential()
#convolutional layer with rectified linear unit activation
model.add(Conv2D(32, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=input_shape))
#again
model.add(Conv2D(64, (5, 5), activation='relu'))
#choose the best features via pooling
model.add(MaxPooling2D(pool_size=(3, 3)))
#randomly turn neurons on and off to improve convergence
model.add(Dropout(0.25))
#flatten since too many dimensions, we only want a classification output
model.add(Flatten())
#fully connected to get all relevant data
model.add(Dense(128, activation='relu'))
#one more dropout for convergence' sake :) 
model.add(Dropout(0.5))
#output a softmax to squash the matrix into output probabilities
model.add(Dense(num_classes, activation='softmax'))
#Adaptive learning rate (adaDelta) is a popular form of gradient descent rivaled only by adam and adagrad
#categorical ce since we have multiple classes (10) 
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
newxs=np.array(pd.read_csv('train.csv').drop(['label'],1))

newys= np.array(pd.read_csv('train.csv')['label'])

       
#train that ish!
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          )
 #how well did it do? 
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