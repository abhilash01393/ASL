# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 18:21:06 2018

@author: Abhilash
"""

import numpy as np

# Import Pandas for data manipulation using dataframes
import pandas as pd
from imutils import paths
# Import Warnings 
import warnings
warnings.filterwarnings('ignore')
from IPython.display import Image

from sklearn.model_selection import train_test_split
from IPython.core.display import HTML 

# Import matplotlib Library for data visualisation
import matplotlib.pyplot as plt

import os
from keras.preprocessing.image import img_to_array
import cv2
import random
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.examples.tutorials.mnist import input_data

# Input data files are available in the "data/" directory.
# For example, running this will list the files in the input directory
#import os
#print(os.listdir("data"))

# construct the argument parse and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-d", "--dataset", 
#	default="./dataset/asl_alphabet_train/train")
##ap.add_argument("-m", "--model", required=True,
##	help="path to output model")
##ap.add_argument("-l", "--labelbin", required=True,
##	help="path to output label binarizer")
##ap.add_argument("-p", "--plot", type=str, default="plot.png",
##	help="path to output accuracy/loss plot")
#args = vars(ap.parse_args())

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 2
INIT_LR = 1e-3
BS = 5
IMAGE_DIMS = (96, 96, 3)
#log_dir = "./logs".format(time())


# grab the image paths and randomly shuffle them
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images("./dataset/asl_alphabet_train/train")))
random.seed(42)
random.shuffle(imagePaths)
 
# initialize the data and labels
data = []
labels = []

# loop over the input images
for imagePath in imagePaths:
	# load the image, pre-process it, and store it in the data list
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
	image = img_to_array(image)
	data.append(image)
 
	# extract set of class labels from the image path and update the
	# labels list
	l = label = imagePath.split(os.path.sep)[-2].split("_")
	labels.append(l)
# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float32") / 255.0

labels = np.array(labels)
print("[INFO] data matrix: {} images ({:.2f}MB)".format(
	len(imagePaths), data.nbytes / (1024 * 1000.0)))

# binarize the labels using scikit-learn's special multi-label
# binarizer implementation
print("[INFO] class labels:")
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)
 
# loop over each of the possible class labels and show them
#for (i, label) in enumerate(mlb.classes_):
#	print("{}. {}".format(i + 1, label))
    
# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.2, random_state=42)
embed_count = 2500
embedding_var = tf.Variable(testX, name='fmnist_embedding')
X_test = testX[:embed_count, 1:] / 255
Y_test = testX[:embed_count, 0]
logdir = 'ASL-logs'
summary_writer = tf.summary.FileWriter(logdir)

# Format: tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto
config = projector.ProjectorConfig()

# You can add multiple embeddings. Here I add only one.
embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name

# Link this tensor to its metadata file (e.g. labels).
embedding.metadata_path = os.path.join(logdir, 'metadata.tsv')

# After constructing the sprite, I need to tell the Embedding Projector where to find it
embedding.sprite.image_path = os.path.join(logdir, 'sprite.png')
embedding.sprite.single_image_dim.extend([96, 96])
projector.visualize_embeddings(summary_writer,config)
with tf.Session() as sesh:
    sesh.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.save(sesh, os.path.join(logdir, 'model.ckpt'))

rows = 28 
cols = 28
label = ['A','B','C']

sprite_dim = int(np.sqrt(X_test.shape[0]))
sprite_image = np.ones((cols * sprite_dim, rows * sprite_dim))

index = 0 
labels = [] 
for i in range(sprite_dim): 
    for j in range(sprite_dim):
        labels.append(label[int(Y_test[index])])

        sprite_image[
            i * cols: (i + 1) * cols,
            j * rows: (j + 1) * rows
        ] = X_test[index].reshape(96, 96) * -1 + 1

        index += 1
        
with open(embedding.metadata_path, 'w') as meta:
    meta.write('Index\tLabel\n')
    for index, label in enumerate(labels):
        meta.write('{}\t{}\n'.format(index, label))
        
plt.imsave(embedding.sprite.image_path, sprite_image, cmap='gray')
plt.imshow(sprite_image, cmap='gray')
plt.show()