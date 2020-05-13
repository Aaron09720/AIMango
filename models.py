# 資料處理套件
import cv2
import csv
import random
import time
import numpy as np
import pandas as pd

# Keras深度學習模組套件
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras import utils as np_utils
from keras import backend as K
from keras import optimizers

# tensorflow深度學習模組套件
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow import keras

from preprocess_data import get_processed_data


# Model for training AIMango model
# 建立深度學習CNN Model
class SampleModel():

    def __init__(self, batch_size=10, epochs=10):
        self.model = tf.keras.Sequential()        

        # 設定超參數HyperParameters 
        self.batch_size =  batch_size
        self.epochs = epochs

        # 取得處理好的測試資料
        self.x_train, self.x_test, self.y_train, self.y_test = get_processed_data()

    def set_layer(self):
        self.model.add(layers.Conv2D(16,(3,3),
                        strides=(1,1),
                        input_shape=(800, 800, 3),
                        padding='same',
                        activation='relu',
                        ))

        self.model.add(layers.BatchNormalization())
        self.model.add(layers.MaxPooling2D(pool_size=(2,2),strides=None))
        self.model.add(layers.Conv2D(32,(3,3),
                        strides=(1,1),
                        padding='same',
                        activation='relu',
                        ))

        self.model.add(layers.MaxPooling2D(pool_size=(2,2),strides=None))
        self.model.add(layers.Conv2D(64,(3,3),
                        strides=(1,1),
                        padding='same',
                        activation='relu',
                        ))

        self.model.add(layers.MaxPooling2D(pool_size=(2,2),strides=None))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64,activation='relu'))
        self.model.add(layers.Dense(128,activation='relu'))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.Dense(3,activation='softmax'))

    def set_optimizer(self):
        adam = optimizers.adam(lr=5)
        self.model.compile(optimizer='adam',
                      loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=['acc'])
    
    def image_enhance(self):
        # zca_whitening 對輸入數據施加ZCA白化
        # rotation_range 數據提升時圖片隨機轉動的角度
        # width_shift_range 圖片寬度的某個比例，數據提升時圖片水平偏移的幅度
        # shear_range 剪切強度（逆時針方向的剪切變換角度）
        # zoom_range 隨機縮放的幅度
        # horizontal_flip 進行隨機水平翻轉
        # fill_mode ‘constant’，‘nearest’，‘reflect’或‘wrap’之一，當進行變換時超出邊界的點將根據本參數給定的方法進行處理

        self.datagen = ImageDataGenerator(
            zca_whitening=False,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

        # 導入圖像增強參數
        self.datagen.fit(self.x_train)
        self.x_train = self.x_train/255
        self.x_test = self.x_test/255
        print('rescale！done!')

    def set_callback(self):
        # 加入EarlyStopping以及Tensorboard等回調函數
        self.CB = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
        # self.TB = keras.callbacks.TensorBoard(log_dir='./log'+"_"+file_name, histogram_freq=1)

    def train_model(self):
        self.set_layer()
        self.image_enhance()
        self.set_callback()
        self.set_optimizer()

        history = self.model.fit(
            x = self.x_train , y = self.y_train,
            batch_size = self.batch_size,
            epochs = self.epochs,
            validation_split = 0.2,
            callbacks = [self.CB]
        )

        print("Complete training model")

    def save_model(self, file_name=None):
        if file_name is None:
            file_name = str(self.batch_size) + '_' + str(self.epochs)

        self.model.save('h5/'+file_name+'.h5')

    def get_model(self):
        return self.model

