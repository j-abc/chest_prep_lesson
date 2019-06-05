import keras
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras import optimizers
import cv2
import os

def get_log_sgd_model(compile_model = True):
    model_name = 'log_sgd'
    
    if compile_model:
        model = Sequential()
        model.add(Flatten(input_shape = (224, 224, 3)))
        model.add(Dense(units = 1, activation = 'sigmoid'))

        model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.95),
                  metrics=['accuracy'])
    else: 
        model = None
    return model, model_name

def get_fcc_sgd_model(compile_model = True):
    model_name = 'fc_sgd'
    if compile_model:    
        model = Sequential()
        model.add(Flatten(input_shape = (224, 224, 3))) 
        model.add(Dense(units = 128, activation = 'relu'))
        model.add(Dense(units = 1, activation = 'sigmoid'))

        model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.95),
                  metrics=['accuracy'])
    else:
        model = None
    return model, model_name

def get_cnn_model(num_layers, dropout, learning_rate, compile_model = True):
    model_name = 'cnn_lay%d_drop%d_lr%d'%(num_layers, dropout, learning_rate)
    
    if compile_model:
        # switch to proper vals
        dropout = dropout/10.0
        learning_rate = 10**(-learning_rate)

        # build model
        model = Sequential()

        model.add(Conv2D(64, (3, 3), input_shape=(224, 224, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        for i in range(num_layers-1):
            model.add(Conv2D(64, (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten()) 

        model.add(Dense(units = 128, activation = 'relu'))

        if dropout > 0:
            model.add(Dropout(dropout))

        model.add(Dense(units = 1, activation = 'sigmoid'))

        # compile
        model.compile(loss='binary_crossentropy',
                      optimizer=optimizers.SGD(lr=learning_rate, momentum=0.95),
                      metrics=['accuracy'])
    else:
        model = None
    return model, model_name

def get_log_model(compile_model = True):
    model_name = 'logistic'
    
    if compile_model:
        model = Sequential()
        model.add(Flatten(input_shape = (224, 224, 3)))
        model.add(Dense(units = 1, activation = 'sigmoid'))

        model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    else: 
        model = None
    return model, model_name

def get_fcc_model(compile_model = True):

    model_name = 'fcc'
    if compile_model:    
        model = Sequential()
        model.add(Flatten(input_shape = (224, 224, 3))) 
        model.add(Dense(units = 128, activation = 'relu'))
        model.add(Dense(units = 1, activation = 'sigmoid'))

        model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    else:
        model = None
    return model, model_name

def get_vgg_model(trainable, compile_model = True):
    if trainable:
        model_name = 'vgg_trainable' 
    else:
        model_name = 'vgg_fixed' 

    if compile_model:
        import numpy as np
        # load vgg 16 
        vgg_conv = VGG16(weights = 'imagenet', 
             include_top = False, 
             input_shape = (224, 224, 3))

        for layer in vgg_conv.layers:
            layer.trainable = trainable

        vgg_model = Sequential()
        out_vgg   = vgg_conv # GlobalAveragePooling2D()(vgg_conv.output)
        vgg_model.add(out_vgg)
        vgg_model.add(GlobalAveragePooling2D())

        vgg_model.add(Dense(1024, activation = 'relu'))
        vgg_model.add(Dropout(0.3))

        vgg_model.add(Dense(512, activation = 'relu'))
        vgg_model.add(Dropout(0.3))

        vgg_model.add(Dense(1, activation = 'sigmoid'))

        vgg_model.compile(loss = 'binary_crossentropy', 
                      optimizer = optimizers.SGD(lr=1e-4, momentum=0.95), 
                      metrics=['accuracy'])

        # https://forums.fast.ai/t/how-to-use-pre-trained-features-from-vgg-16-as-input-to-globalaveragepooling2d-layer-in-keras/3196/3
    else:
        vgg_model = None
    return vgg_model, model_name
