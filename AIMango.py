# 資料處理套件
import os
import cv2
import csv
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # plt 用於顯示圖片

# Keras深度學習模組套件
import keras
from keras.preprocessing.image import img_to_array, load_img

from models import SampleModel
from preprocess_data import get_processed_data


''' 繪製學習成效 '''
def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()

''' 推測圖片 '''
def predict(model):
    x_train, x_test, y_train, y_test = get_processed_data()
    

    test_mango_dir = os.path.join("AIMango/test_image")
    test_mango_fnames = os.listdir(test_mango_dir)

    img_files = [os.path.join(test_mango_dir,f) for f in test_mango_fnames]
    img_path = random.choice(img_files)

    # 讀入待測試圖像並秀出
    img = load_img(img_path, target_size=(800, 800))  # this is a PIL image

    labels = ['等級A','等級B',"等級C"]

    # 將圖像轉成模型可分析格式(800x800x3, float32)
    x = img_to_array(img)  # Numpy array with shape (800, 800, 3)
    x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 800, 800, 3)
    x /= 255 # Rescale by 1/255

    start = time.time() # 啟動計時器
    result = model.predict(x) # 對輸入圖像進行推論(預測)
    finish = time.time() # 結束計時器

    pred = result.argmax(axis=1)[0]
    pred_prob = result[0][pred]

    print("Result = %f" %pred_prob) # 印出結果可能機率值(0.0 ~ 1.0)
    print("Test time :%f second." %(finish-start)) # 印出推論時間

    # 設定分類門檻值並印出推論結果
    print("有 {:.2f}% 機率為{}".format(pred_prob * 100,labels[pred])) # 印出推論時間


    ''' 測試集預測準確度 '''
    # 測試集標籤預測
    y_pred = model.predict(x_test)

    # 整體準確度
    count = 0
    for i in range(len(y_pred)):
        if(np.argmax(y_pred[i]) == np.argmax(y_test[i])): #argmax函数找到最大值的索引，即为其类别
            count += 1
    score = count/len(y_pred)
    print('正確率為:%.2f%s' % (score*100,'%'))

if __name__ == "__main__":
    sample_model = SampleModel()
    sample_model.train_model()
    predict(sample_model.get_model())


# 模型預測後的標籤
'''
predict_label = np.argmax(y_pred,axis=1)
print(predict_label)
print(len(predict_label))
'''

# 模型原標籤
'''
true_label = y_label_org[84:]
true_label = np.array(true_label)
print(true_label)
print(len(true_label))
'''

# 模型預測後的標籤
'''
predictions = model.predict_classes(x_test)
print(predictions)
print(len(predictions))

pd.crosstab(true_label,predict_label,rownames=['實際值'],colnames=['預測值'])
'''
