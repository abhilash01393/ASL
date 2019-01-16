# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 16:52:58 2018

@author: Abhilash
"""

from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
#from convnet_drawer import Model, Conv2D, MaxPooling2D, Flatten, Dense
#from pptx_util import save_model_to_pptx
#from matplotlib_util import save_model_to_file

model = Sequential()
height = 96
width = 96
depth = 3
inputShape = (height, width, depth)
chanDim = -1
classes = 3
finalAct = "softmax"
 
		# if we are using "channels first", update the input shape
		# and channels dimension
if K.image_data_format() == "channels_first":
    inputShape = (depth, height, width)
    chanDim = 1
            
        # CONV => RELU => POOL
model.add(Conv2D(64, (3, 3), padding="same",
			input_shape=inputShape))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))
        
        # (CONV => RELU) * 2 => POOL
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
 
		# (CONV => RELU) * 2 => POOL
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
        
        # first (and only) set of FC => RELU layers
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
 
		# use a *softmax* activation for single-label classification
		# and *sigmoid* activation for multi-label classification
model.add(Dense(classes))
model.add(Activation(finalAct, name = "predictions"))
        



# save as svg file
model.save_fig("example.svg")

# save as pptx file
#save_model_to_pptx(model, "example.pptx")
#
## save via matplotlib
#save_model_to_file(model, "example.pdf")