# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import nibabel as nib
import glob
import os
import pandas as pd
import csv as csv
import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten,Reshape
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D,AveragePooling3D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, merge,Lambda, Embedding, Bidirectional, LSTM, Dense, RepeatVector, Dropout,Activation,BatchNormalization
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import scipy
from keras.callbacks import History
from keras.regularizers import l2
from skimage import transform
from scipy.ndimage import zoom
from keras.layers.wrappers import TimeDistributed
from keras.layers import dot,concatenate
from self_Attention_keras import Self_Attention
print (keras.backend.image_data_format())
import warnings
warnings.filterwarnings("ignore")
if True:
    files_all = sorted(glob.glob("nii"))

    df = pd.read_csv('csv')
    y = df.values # A numpy array containing age label of 559 persons
    #print ('\n here \n',y.shape)
    #print ('\n here \n',len(files_all))
    y=y[:,1]  
    # y= y.astype(int)
    #train_files, test_files, train_labels, test_labels = train_test_split(files_all,y,test_size=0.2)

    print("--------files_all----------",len(files_all))


    test_files =sorted(glob.glob("nii"))
    # test_files = glob.glob("E:/Data-IXI/IXI-TEST-P/*.nii")
    dft = pd.read_csv('csv')
    yt = dft.values
    yt = yt[:,1]
    # yt= yt.astype(int)
    # test_labels =yt

    print("------test_files----------",len(test_files))
   # np.random.seed(200)
   # np.random.shuffle(files_all)
   # np.random.seed(200)
   # np.random.shuffle(y)
   # np.random.seed(200)
   # np.random.shuffle(test_files)



batch_size =16
num_classes = 100
epochs = 100
file_size = 110
weight_decay = 0.00005
dimx,dimy,channels = 121,145,121
#0.0001)0.001

# Convert class vectors to binary class matrices.
inpx = Input(shape=(dimx,dimy,channels,1),name='inpx')


#keer------------------------------------------------------------
#,kernel_regularizer=l2(weight_decay)
x = Convolution3D(8, 3, 3, 3, padding='same',name='conv1')(inpx)
x = Activation('elu')(x)
x = Convolution3D(8, 3, 3, 3, padding='same',name='conv2')(x)
x = Activation('elu')(x)
x = MaxPooling3D(pool_size=(2, 2, 2),strides=(2,2,2),padding='same', name='pool1')(x)
x = BatchNormalization()(x)
#x = Dropout(0.5)(x)
x = Convolution3D(16, 3, 3, 3, padding='same',name='conv3')(x)
x = Activation('elu')(x)
x = Convolution3D(16, 3, 3, 3, padding='same',name='conv4')(x)
x = Activation('elu')(x)
x = MaxPooling3D(pool_size=(2, 2, 2),strides=(2,2,2),padding='same', name='pool2')(x)
x = BatchNormalization()(x)
x = Convolution3D(32, 3, 3, 3, padding='same',name='conv5')(x)
x = Activation('elu')(x)
x = Convolution3D(32, 3, 3, 3, padding='same',name='conv6')(x)
x = Activation('elu')(x)
x = MaxPooling3D(pool_size=(2, 2, 2),strides=(2,2,2),padding='same', name='pool3')(x)
x = BatchNormalization()(x)
#x = Dropout(0.5)(x)
x = Convolution3D(64, 3, 3, 3, padding='same',name='conv7')(x)
x = Activation('elu')(x)
x = Convolution3D(64, 3, 3, 3, padding='same',name='conv8')(x)
x = Activation('elu')(x)
x = MaxPooling3D(pool_size=(2, 2, 2),strides=(2,2,2),padding='same', name='pool4')(x)
x = BatchNormalization()(x)
#x = Dropout(0.5)(x)
# x = Convolution3D(128, 3, 3, 3, padding='same',name='conv9')(x)
# x = Activation('relu')(x)
# x = Convolution3D(128, 3, 3, 3, padding='same',name='conv10')(x)
# x = BatchNormalization()(x)
# x = Activation('relu')(x)
# x = MaxPooling3D(pool_size=(2, 2, 2),strides=(2,2,2),padding='same', name='pool5')(x)
#x = Dropout(0.5)(x)

#add seld-attention
#x = Reshape((80, 128))(x)
#x = Self_Attention(128)(x)
# x=Dropout(0.5)(x)


#add attention
#x = Reshape((80, 128))(x)
#att_1 = Activation('tanh')(x)
#att_1 = TimeDistributed(Dense(1))(att_1)
#att_1 = Activation('softmax')(att_1)
#print("--------------att_1-",att_1.shape)
#att_1_pro = Flatten()(att_1)
#print("=======f====",att_1_pro.shape)
#img_new = dot([att_1_pro, x], axes=(1, 1))
#score = Dense(100, activation='softmax', name='fc8')(img_new)
#add attention

hx = Flatten()(x)

#hx = Dense(512,activation='relu',name='fc7',)(hx)
# hx = Dense(1024,activation='relu',name='fc8')(hx)
hx = Dense(256,activation='elu',name='fc7')(hx)
score = Dense(1,name='fc9')(hx)
model = Model(inputs=inpx, outputs=score)
opt=keras.optimizers.Adam(lr=0.00001,beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#opt = keras.optimizers.rmsprop(lr=0.01, decay=1e-6) 
#opt = keras.optimizers.sgd(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#opt = keras.optimizers.SGD(lr=0.00005,decay=1e-6, momentum=0.9, nesterov=True)
#opt=keras.optimizers.Adam(lr=0.000001,beta_1=0.9, beta_2=0.999, epsilon=1e-08,amsgrad=True)
#keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedula'dae_decay=0.004)
#model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

model.compile(loss='mse', optimizer=opt, metrics=['mae'])
#x_train = np.expand_dims(x_train,axis=0)
#x_train = np.reshape(x_train,(8,256,256,144,1))

model.summary()
#file_size = 10
#for i in range(len(files_all)/file_size):
for i in range(1):
    ##files = files_all[i*file_size:(i+1)*file_size]#
    files = files_all[:]
    train_x = []
    for f in files:
        img = nib.load(f)
        img_data = img.get_data()
        img_data = np.asarray(img_data) 
        img_data = img_data.astype('float32')
        img_data = img_data /255
        #print("---------",img_data.shape)
        #200,215,180
        # if(img_data.shape==(176,256,256)):
        #     img_data = img_data.reshape([256,256,176])
        ##img_data = img_data[:, :, 0:144]
        ##img_data = img_data[0:65,0:65,0:55]
        #img_data = zoom(img_data,(0.54,0.45,0.46),order=3,mode='constant',cval=0.0,prefilter=True)
        # img_data = transform.resize(img_data,(65,65,55))
        # img_data = img_data*255
        # img_data = img_data.astype(np.uint8)
        train_x.append(img_data)
    #x_train = train_x[:-2]
    x_train = np.asarray(train_x)
    print('\n iteration number :', i,'\n')
    #x_train = x_train.astype('float32')
    #x_train /= 255
    x_train = np.expand_dims(x_train,4) 
    print ('\n', x_train.shape)
    ##y_train = y[i*file_size:(i+1)*file_size]
    y_train = y[:]

test_x, test_y = [], yt
test_yv = yt
for i, f in enumerate(test_files):
    # for i,f in enumerate(test_files[0:10]):
    img = nib.load(f)
    img_data = img.get_data()
    img_data = np.asarray(img_data)
    img_data = img_data.astype('float32')
    img_data = img_data / 255
    test_x.append(img_data)

test_x = np.asarray(test_x)
test_x = np.expand_dims(test_x, 4)

#k折交叉验证
k=10
num_val_samples = len(x_train)//k
print("num_val_samples",num_val_samples)
num_epochs = 100
all_scores = []

for i in range(k):
    print('processing fold #',i)
    val_data = x_train[i*num_val_samples:(i+1)*num_val_samples]
    val_targets = y_train[i*num_val_samples:(i+1)*num_val_samples]

    partial_train_data = np.concatenate([x_train[:i*num_val_samples],
                                         x_train[(i+1)*num_val_samples:]],axis=0)
    partial_train_targets = np.concatenate([y[:i * num_val_samples],
                                         y[(i + 1) * num_val_samples:]], axis=0)
    print("----------------len(train)", len(partial_train_data))
    print("----------------len(val)", len(val_data))
    # for j in range(num_epochs):
    #
    #     history = History()
    model.fit(partial_train_data, partial_train_targets, batch_size=batch_size, epochs=100, verbose=2)
    #     #model.save_weights('my_model_weights.h5')
    #     y_pred = model.predict(val_data, batch_size=batch_size, verbose=2)
    #     #y_pred = np.argsort(y_pred, axis=1)
    #     MAE = mean_absolute_error(val_targets, y_pred)
    #     print("Epoch:", (j + 1), "-----------------------test_mae--------------------:", MAE)
    mse, mae = model.evaluate(val_data, val_targets)
    print('-------mse-----mae--------------', mse, mae)
    all_scores.append(mae)



print("------all_scores----------",all_scores)
print("-----mean----------",np.mean(all_scores))


mae=10240
for j in range(700):

    history = History()

    # input_x = [x,img_x]#type is list

    model.fit(x_train, y_train, batch_size=batch_size, epochs=1, verbose=2)
    #
    # model.save_weights('my_model_weights.h5')
    y_pred = model.predict(test_x, batch_size=batch_size, verbose=2)
    #y_pred = np.argsort(y_pred, axis=1)
    MAE = mean_absolute_error(yt, y_pred)
    if MAE < mae:
        mae = MAE
        model.save('3dcnn-k_R.h5')
    print("Epoch:", (j + 1), "-----------------------test_mae--------------------:", MAE)




# model.fit(x_train, y_train,
#          batch_size=batch_size,
#          epochs=500,verbose=2)


Pred = model.predict([test_x])

print("--------turelabels---------",test_yv)
print("------predlabes---",Pred)


#test_y = [i[0]for i in test_y[0:len(pred)]]
#print (test_y[0:10])

#
MAE = mean_absolute_error(yt,Pred)
# ##pearson = scipy.stats.pearsonr(test_y,Pred)
r2 = r2_score(test_y,Pred)
# mse = mean_squared_error(test_labels,Pred)
# ##scores =[mae,pearson,r2,mse]
# scores =[MAE,r2,mse]
#
# pd.to_pickle(scores,'E:/DataSet/Data-IXI-5/scores_out')
# pd.to_pickle(pred,'E:/DataSet/Data-IXI-5/pred_out')

# print("--Pred--",pred)
# print("--test labels--",test_labels)



print('\n\n MAE is : ', MAE)
# ##print('\n\n pearsonr is:-',pearson)
print('\n\n R2 is : ', r2)
# print('\n\n MSE is : ', mse)
# print('\n\n')

#model.save('3dcnn-k.h5')