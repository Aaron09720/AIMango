import csv
import cv2
import numpy as np
import pandas as pd

import tensorflow as tf

from keras import utils as np_utils
from keras.preprocessing.image import img_to_array

def get_processed_data():

    # 讀取資料集標籤檔
    # Sample_label = pd.read_csv("AIMango/Sample_Label.csv",encoding="utf8")
    csvfile = open('AIMango/Sample_Label.csv')
    reader = csv.reader(csvfile)

    # 讀取csv標籤
    labels = []
    for line in reader:
        tmp = [line[0],line[1]]
        # print tmp
        labels.append(tmp)

    csvfile.close() 

    X = []
    y = []

    # 轉換圖片的標籤
    for i in range(len(labels)):
        labels[i][1] = labels[i][1].replace("等級A","0")
        labels[i][1] = labels[i][1].replace("等級B","1")
        labels[i][1] = labels[i][1].replace("等級C","2")

    # 隨機讀取圖片
    a = 0
    items= []

    import random
    for a in range(0,94):
        items.append(a)

    # 製作訓練用資料集及標籤
    for i in random.sample(items,94):
        img = cv2.imread("AIMango/sample_image/" + labels[i][0])
        res = cv2.resize(img,(800,800),interpolation=cv2.INTER_LINEAR)
        res = img_to_array(res)
        X.append(res)
        y.append(labels[i][1])

    # 轉換至array的格式
    X = np.array(X)
    y = np.array(y)

    # 轉換至float的格式
    for i in range(len(X)):
        X[i] = X[i].astype('float32')

    # 將標籤轉換至float格式
    y = tf.strings.to_number(y, out_type=tf.float32)

    # 標籤進行one-hotencoding
    y = np_utils.to_categorical(y, num_classes = 3)

    ''' 製作訓練資料集 '''
    # 分配訓練集及測試集比例
    x_train = X[:84]
    y_train = y[:84]
    x_test = X[84:]
    y_test = y[84:]

    return x_train, x_test, y_train, y_test
    