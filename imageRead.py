# -*- coding: utf-8 -*-
import cv2
from imutils import paths
from keras.preprocessing.image import img_to_array
import os
import numpy as np
import matplotlib

data = []
labels = []
IMAGE_DIMS = (96, 96, 3)
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images("./testdata")))
for imagePath in imagePaths:
	# load the image, pre-process it, and store it in the data list
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
	image = img_to_array(image)
	data.append(image)
 
	# extract the class label from the image path and update the
	# labels list
	label = imagePath.split(os.path.sep)[-2]
	labels.append(label)

    
data = np.array(data, dtype="float") / 255.0
print(data)
labels = np.array(labels)
print("[INFO] data matrix: {:.2f}MB".format(
	data.nbytes / (1024 * 1000.0)))