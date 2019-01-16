# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 13:36:12 2018

@author: Abhilash
"""

# set the matplotlib backend so figures can be saved in the background

# import the necessary packages

from __future__ import print_function

from keras.preprocessing.image import ImageDataGenerator
from time import time
from os.path import exists, join
from os import makedirs
from keras import backend as K
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping, CSVLogger
from tensorresponseboard import TensorResponseBoard as TRB
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from SmallerVGGNet import SmallerVGGNet
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", 
	default="./dataset/asl_alphabet_train/train")
#ap.add_argument("-m", "--model", required=True,
#	help="path to output model")
#ap.add_argument("-l", "--labelbin", required=True,
#	help="path to output label binarizer")
#ap.add_argument("-p", "--plot", type=str, default="plot.png",
#	help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 30
INIT_LR = 1e-3
BS = 5
IMAGE_DIMS = (96, 96, 3)
log_dir = "./logs"


# grab the image paths and randomly shuffle them
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(args["dataset"])))
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
for (i, label) in enumerate(mlb.classes_):
	print("{}. {}".format(i + 1, label))
    
# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.2, random_state=42)
print(testX.shape)
with open(join(log_dir, 'metadata.tsv'), 'w') as f:
    np.savetxt(f, testY)
plt.figure()
plt.figure(figsize=[5,5])

# Display the first image in training data
plt.subplot(121)
plt.imshow(trainX[0,:,:], cmap='gray')
plt.title("Ground Truth : {}".format(trainY[0]))

# Display the first image in testing data
plt.subplot(122)
plt.imshow(testX[0,:,:], cmap='gray')
plt.title("Ground Truth : {}".format(testY[0]))
# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")
K.set_learning_phase(1) #set learning phase
# initialize the model using a sigmoid activation as the final layer
# in the network so we can perform multi-label classification
print("[INFO] compiling model...")
model = SmallerVGGNet.build(
	width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
	depth=IMAGE_DIMS[2], classes=len(mlb.classes_),
	finalAct="sigmoid")
 
# initialize the optimizer
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
#tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
tensorboard = TensorBoard(log_dir="logs/{}".format(time()), histogram_freq=1, write_graph=True, write_images=True)
embedding_layer_names = set(layer.name
                            for layer in model.layers
                            if layer.name.startswith('con_2d'))
print(embedding_layer_names)

tensorboard1 = TensorBoard(log_dir=log_dir, batch_size=BS,
                          embeddings_freq=1,
                          embeddings_layer_names=['features'],
                          embeddings_metadata='metadata.tsv',
                          embeddings_data=testX)

earlyStopImprovement = EarlyStopping(monitor='val_acc', min_delta=0.01, patience=4, verbose=1, mode='auto')
modelCheckPoint = ModelCheckpoint("./new_models/weights-{epoch:02d}.model", monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
log = CSVLogger('./log/log.csv')


embedding_layer_names = set(layer.name
                            for layer in model.layers
                            if layer.name.startswith('dense_'))
print(embedding_layer_names)
#tb = TensorBoard(log_dir="logs/{}".format(time()), histogram_freq=1, batch_size=BS,
#                           write_graph=True, write_grads=True, write_images=True,
#                           embeddings_freq=10, embeddings_metadata=None,
#                           embeddings_layer_names=embedding_layer_names)
#tb = TRB(log_dir="logs/{}".format(time()), histogram_freq=10, batch_size=BS,
#                         write_graph=True, write_grads=True, write_images=True,
#                         embeddings_freq=10,
#                         embeddings_layer_names=embedding_layer_names,
#                         embeddings_metadata='metadata.tsv',
#                         val_size=len(testX), img_path='images.jpg', img_size=[28, 28])

#from PIL import Image
#img_array = testX.reshape(31, 96, 96, 3)
#img_array_flat = np.concatenate([np.concatenate([x for x in row], axis=1) for row in img_array])
#img = Image.fromarray(np.uint8(255 * (1. - img_array_flat)))
#img.save(os.path.join(log_dir, 'images.jpg'))
#np.savetxt(os.path.join(log_dir, 'metadata.tsv'), np.where(testY)[1], fmt='%d')

#metadata_file = os.path.join(log_dir, 'metadata.tsv')
#with open(metadata_file, 'w') as f:
#    for i in range(len(testY)):
#        c = np.nonzero(testY[i])[0][0]
#        f.write('{}\n'.format(c))
# compile the model using binary cross-entropy rather than
# categorical cross-entropy -- this may seem counterintuitive for
# multi-label classification, but keep in mind that the goal here
# is to treat each output label as an independent Bernoulli
# distribution
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])
f = open("lb.pickle", "wb")
f.write(pickle.dumps(mlb))
f.close()
# train the network
print("[INFO] training network...")
H = model.fit_generator(
	aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY),
	steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1, callbacks=[tensorboard, modelCheckPoint,log])

# save the model to disk
print("[INFO] serializing network...")
model.save("training.model")
#summarize layers
model.summary()
# plot graph
plot_model(model, to_file='convolutional_neural_network.png')
# save the multi-label binarizer to disk
print("[INFO] serializing label binarizer...")
f = open("lb.pickle", "wb")
f.write(pickle.dumps(mlb))
f.close()

#import tensorflow as tf
#from tensorflow.contrib.tensorboard.plugins import projector
#emb = model.predict(testX)
#embedding_var = tf.Variable(emb,  name='final_layer_embedding')
#sess = tf.Session()
#sess.run(embedding_var.initializer)
#summary_writer = tf.summary.FileWriter(log_dir)
#config = projector.ProjectorConfig()
#embedding = config.embeddings.add()
#embedding.tensor_name = embedding_var.name
#
## Specify the metadata file:
#embedding.metadata_path = os.path.join(log_dir, 'metadata.tsv')
#
## Specify the sprite image: 
#embedding.sprite.image_path = os.path.join(log_dir, 'mnist_10k_sprite.png')
#embedding.sprite.single_image_dim.extend([96, 96]) # image size = 28x28
#
#projector.visualize_embeddings(summary_writer, config)
#saver = tf.train.Saver([embedding_var])
#saver.save(sess, os.path.join(log_dir, 'model2.ckpt'), 1)

#Model Evaluation on the Test Set
test_eval = model.evaluate(testX, testY, verbose=0)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])
# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig("plot.png")

#Predict Labels
predicted_classes = model.predict(testX)
#print(predicted_classes)
predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
test_Y_one_hot = np.argmax(np.round(testY),axis=1)
print(predicted_classes.shape, test_Y_one_hot.shape)
#print(predicted_classes)
correct = np.where(predicted_classes==test_Y_one_hot)[0]
#test_Y_one_hot = np.argmax(np.round(testY),axis=1)
#print(testY)
#print(correct)
#print(test_Y_one_hot)
plt.figure(figsize=(15,15))
print("Found %d correct labels" %len(correct))
for i, correct in enumerate(correct[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(testX[correct,:,:], cmap='gray', interpolation='none')
#    plt.imshow(testX[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(mlb.classes_[predicted_classes[correct]], mlb.classes_[test_Y_one_hot[correct]]))
#    plt.constrained_layout()
plt.savefig('correct.jpg')

incorrect = np.where(predicted_classes!=test_Y_one_hot)[0]
#test_Y_one_hot = np.argmax(np.round(testY),axis=1)
#print(testY)
#print(correct)
#print(test_Y_one_hot)
#print(incorrect)
plt.figure(figsize=(15,15))
print("Found %d incorrect labels" %len(incorrect))
for i, incorrect in enumerate(incorrect[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(testX[incorrect,:,:], cmap='gray', interpolation='none')
#    plt.imshow(testX[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(mlb.classes_[predicted_classes[incorrect]], mlb.classes_[test_Y_one_hot[incorrect]]))
#    plt.constrained_layout()
plt.savefig('incorrect.jpg')
#Classification Report
from sklearn.metrics import classification_report
target_names = ["Class {}".format(mlb.classes_[i]) for i in range(len(mlb.classes_))]
print(classification_report(test_Y_one_hot, predicted_classes, target_names=target_names))

#ROC Curve
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt1
from itertools import cycle
from sklearn.metrics import roc_curve, auc

# Plot linewidth.
lw = 2

# Compute ROC curve and ROC area for each class
n_classes = len(mlb.classes_)
y_score = model.predict(testX)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(testY[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(testY.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt1.figure()
plt1.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt1.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt1.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt1.plot([0, 1], [0, 1], 'k--', lw=lw)
plt1.xlim([0.0, 1.0])
plt1.ylim([0.0, 1.05])
plt1.xlabel('False Positive Rate')
plt1.ylabel('True Positive Rate')
plt1.title('Some extension of Receiver operating characteristic to multi-class')
plt1.legend(loc="lower right")
plt1.savefig('roc1.png')


# Zoom in view of the upper left corner.
plt1.figure()
plt1.xlim(0, 0.2)
plt1.ylim(0.8, 1)
plt1.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt1.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt1.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt1.plot([0, 1], [0, 1], 'k--', lw=lw)
plt1.xlabel('False Positive Rate')
plt1.ylabel('True Positive Rate')
plt1.title('Some extension of Receiver operating characteristic to multi-class')
plt1.legend(loc="lower right")
plt1.savefig('roc2.png')

#loss and accuracy curves
plt.figure()
plt.plot(H.history['acc'],'r')
plt.plot(H.history['val_acc'],'g')
plt.xticks(np.arange(0, EPOCHS, 2.0))
plt.rcParams['figure.figsize'] = (8, 6)
plt.xlabel("Num of Epochs")
plt.ylabel("Accuracy")
plt.title("Training Accuracy vs Validation Accuracy")
plt.legend(['train','validation'])
plt.savefig("Accuracy.jpg")
 
plt.figure()
plt.plot(H.history['loss'],'r')
plt.plot(H.history['val_loss'],'g')
plt.xticks(np.arange(0, EPOCHS, 2.0))
plt.rcParams['figure.figsize'] = (8, 6)
plt.xlabel("Num of Epochs")
plt.ylabel("Loss")
plt.title("Training Loss vs Validation Loss")
plt.legend(['train','validation'])
plt.savefig("Loss.jpg")
 
plt.show()

#Model Accuracy
scores = model.evaluate(testX, testY, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# Confusion matrix result
num_classes=len(mlb.classes_)
from sklearn.metrics import classification_report, confusion_matrix
Y_pred = model.predict(testX, verbose=2)
y_pred = np.argmax(Y_pred, axis=1)

for ix in range(num_classes):
    print(ix, confusion_matrix(np.argmax(testY,axis=1),y_pred)[ix].sum())
cm = confusion_matrix(np.argmax(testY,axis=1),y_pred)
print(cm)

# Visualizing of confusion matrix
import seaborn as sn
import pandas  as pd


df_cm = pd.DataFrame(cm, range(num_classes),
                  range(num_classes))
plt.figure(figsize = (15,15))
sn.set(font_scale=1.4)#for label size
sns_fig = sn.heatmap(df_cm, annot=True,annot_kws={"size": 12})# font size
sns_fig.figure.savefig("Confusion Matrix.jpg")
plt.show()

#Visualize CNN Layers
plt.figure()
from keras.models import Model
layer_outputs = [layer.output for layer in model.layers]
activation_model = Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(trainX[10].reshape(1,96,96,3))
print(len(activations))
def display_activation(activations, col_size, row_size, act_index): 
    activation = activations[act_index]
    activation_index=0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
    for row in range(0,row_size):
        for col in range(0,col_size):
            ax[row][col].imshow(activation[0, :, :, activation_index])
            activation_index += 1
    plt.savefig('Activation Layer %d.png' % act_index)
for i in range(0,len(activations)-1,9):
    display_activation(activations, 8, 8, i)

#GIF
#from vis.losses import ActivationMaximization
#from vis.regularizers import TotalVariation, LPNorm
#from vis.input_modifiers import Jitter
#from vis.optimizer import Optimizer
#
#from vis.callbacks import GifGenerator
#from keras.applications.vgg16 import VGG16
#
## Build the VGG16 network with ImageNet weights
#
## The name of the layer we want to visualize
## (see model definition in vggnet.py)
#layer_name = 'predictions'
#layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
#output_class = [1]
#
#losses = [
#    (ActivationMaximization(layer_dict[layer_name], output_class), 2),
#    (LPNorm(model.input), 10),
#    (TotalVariation(model.input), 10)
#]
#opt = Optimizer(model.input, losses)
#opt.minimize(max_iter=500, verbose=True, image_modifiers=[Jitter()], callbacks=[GifGenerator('opt_progress')])