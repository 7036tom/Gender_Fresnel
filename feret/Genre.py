# coding: utf-8

import PIL
import Image
import numpy
import ImageOps
import sys
import os

# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.models import load_model

from sklearn.model_selection import StratifiedKFold

import numpy
import pandas
from keras.regularizers import l1, activity_l1
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

from keras.utils import np_utils

dataframe = pandas.read_csv("Database1010feret", header=None, sep=" ")
dataset = dataframe.values
# split into input (X) and output (Y) variables

X = dataset[0:2580,0:100]
Y = dataset[0:2580:,100]
X_test = dataset[2580:2600,0:100]
Y_test = dataset[2580:2600:,100]
Y_test = np_utils.to_categorical(Y_test, 2)

model = load_model('model_naif')

usr_input = 0

while (usr_input != 1):
	path = raw_input("Veuillez entrer le chemin de votre image, si possible plutot carrée")
	path = path[1:len(path)-2]
	img = Image.open(path, 'r').resize((10,10))

	# Passe l'image en niveau de gris
	img = ImageOps.grayscale(img)

	# img -> data
	imgdata = img.getdata()
	image_tab = numpy.array(imgdata)
	#print(image_tab[0:99])
	X_test[0] = image_tab[0:100]
	
	Z = model.predict(X_test, batch_size=32, verbose=0)
	print(Z[0])


	

