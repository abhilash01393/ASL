# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 19:49:26 2018

@author: Abhilash
"""

import os,jpg,array
import pandas as pd
os.chdir('./dataset/asl_alphabet_train/train')
import time

from PIL import Image
columnNames = list()

for i in range(784):
    pixel = 'pixel'
    pixel += str(i)
    columnNames.append(pixel)


train_data = pd.DataFrame(columns = columnNames)
start_time = time.time()
for i in range(1,49000):
    t = i
    img_name = str(t)+'.jpg'
    img = Image.open(img_name)
    rawData = img.load()
        #print rawData
    data = []
    for y in range(28):
        for x in range(28):
            data.append(rawData[x,y][0])
    print(i)
    k = 0
        #print data
    train_data.loc[i] = [data[k] for k in range(784)]
    #print train_data.loc[0]

print("Done")
print(time.time()-start_time)

#os.chdir('../../')
#label_data = pd.read_csv("train.csv")
#print label_data
#train_labels = label_data['label']
#print label_data['label']
#train_data = pd.concat([train_data,label_data],axis = 1)
#train_data = train_data.drop('filename',1)
print(train_data)

train_data.to_csv("train_converted.csv",index = False)
print("Done1")
print(time.time()-start_time)