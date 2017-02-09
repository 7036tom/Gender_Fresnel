# coding: utf-8

# import the necessary packages for photo handling

import PIL
import Image
import numpy
import ImageOps
import sys
import os

# import the necessary packages for cam handling
import argparse
import datetime
import imutils
import time
import cv2

cascPath = 'haarcascade_frontalface_default.xml'
# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.models import load_model
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import StratifiedKFold

import numpy
import pandas
from keras import backend as K
from keras.regularizers import l1l2
K.set_image_dim_ordering('th') # Avoid input to Flatten is not fully defined

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

from keras.utils import np_utils

# load train dataset
dataframe = pandas.read_csv("Database2828cropped3.0", header=None, sep=" ")
dataset = dataframe.values
# load train dataset
dataframe_test = pandas.read_csv("BaseTest2828", header=None, sep=" ")
dataset_test = dataframe_test.values
# split into input (X) and output (Y) variables
# Train
X = dataset[0:5302,0:784]
Y = dataset[0:5302:,784]
# Test
X_test = dataset_test[0:965,0:784]
Y_test = dataset_test[0:965:,784]

Y_test = np_utils.to_categorical(Y_test, 2)
#Y = np_utils.to_categorical(Y, 2) # convert class vectors to binary class matrices
#Y2 = np_utils.to_categorical(Y2, 2)

# reshape to be [samples][pixels][width][height]
X = X.reshape(X.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')


# normalize inputs from 0-255 to 0-1
X= X / 255
X_test = X_test / 255

#Define model

model = Sequential()
model.add(Convolution2D(64, 4, 4, border_mode='valid', input_shape=(1, 28, 28), init='he_normal', activation='relu', W_regularizer=l1l2(l1=0.000005, l2=0.00005)))
model.add(MaxPooling2D(pool_size=(2, 2)))  
model.add(Dropout(0.4))
model.add(Convolution2D(128, 8, 8, activation='relu', init='he_normal'))
model.add(MaxPooling2D(pool_size=(2, 2))) # Sous echantillonage 1/2
model.add(Dropout(0.4))




model.add(Flatten())
model.add(Dense(300, activation ='relu', init='he_normal'))#, W_regularizer=l1l2(l1=0.000005, l2=0.00005)))
model.add(Dropout(0.6))
model.add(Dense(2))
model.add(Activation('softmax'))


"""
model.add(Dense(60, input_dim=99, init='normal', activation='relu' ,W_regularizer=l1(0.000001), activity_regularizer=activity_l1(0.000001)))
model.add(Dropout(0.4))
model.add(Dense(60, activation='relu' ))
model.add(Dropout(0.4))
model.add(Dense(60, activation ='relu'))
model.add(Dropout(0.4))
model.add(Dense(2))
model.add(Activation('softmax'))
"""
#model.add(Activation('sigmoid'))



# Compile model
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#sgd = SGD(lr=0.01, momentum=0, decay=0, nesterov=False)
rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.000)
model.compile(optimizer=rms, loss='binary_crossentropy', metrics=['accuracy']) # Gradient descent
#model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy']) # Gradient descent
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # Gradient descent

# Affiche les details du reseau !
print (model.summary()) 

# Weights more adapted to the class imbalanced of the issue.
class_weight = {0 : 0.4,
    1: 0.6}

# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
cvscores = []
for train, test in kfold.split(X, Y):
	# create model
	
	Y3 = np_utils.to_categorical(Y, 2) # convert class vectors to binary class matrices
	
	
	# Early stopping.
		
	callbacks = [
    EarlyStopping(monitor='val_loss', patience=30, verbose=0),
    ModelCheckpoint("/home/tom/Documents/Projets python/bosongit/weigh.hdf", monitor='val_loss', save_best_only=True, verbose=0),
	]
		
		
	# Fit the model		
	model.fit(X[train], Y3[train],validation_data=(X_test, Y_test), class_weight=class_weight, nb_epoch=200, batch_size=20, shuffle=True, verbose=1, callbacks=callbacks)#, class_weight=class_weight)
	

	"""
	# Fit the model

	model.fit(X[train], Y3[train], nb_epoch=100, batch_size=96)
	"""

	# evaluate the model
	scores = model.evaluate(X[test], Y3[test])
	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
	cvscores.append(scores[1] * 100)

print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))

model.save("model_Convo")

#print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#test

#scores = model.evaluate(X, Y)
#print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#Z = model.predict(X_test, batch_size=32, verbose=0)
#print(Z[1:10])
#print(Y_test[1:10])



	

	

	

	

