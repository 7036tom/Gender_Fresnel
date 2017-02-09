# coding: utf-8

import PIL
import Image
import numpy
import ImageOps
import sys
import os
import imutils
import cv2

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
from keras import backend as K
K.set_image_dim_ordering('th') # Avoid input to Flatten is not fully defined
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

from keras.utils import np_utils

# Create the haar cascade
cascPath = 'haarcascade_frontalface_default.xml'
#cascPath = 'haarcascade_frontalface_alt.xml'


# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

dataframe = pandas.read_csv("Database2828cropped2.0", header=None, sep=" ")
dataset = dataframe.values
# split into input (X) and output (Y) variables


X_test = dataset[2580:2600,0:784]


model = load_model('model_Convo')

usr_input = 0

while (usr_input != 1):
	X_test = dataset[2580:2600,0:784]
	path = raw_input("Veuillez entrer le chemin de votre image, si possible plutot carrée")
	path = path[1:len(path)-2]
	
	# Read the image
	img = cv2.imread(path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # On passe de BGR a RGB !

	# Detect faces in the image
	faces = faceCascade.detectMultiScale(img, 1.1,  5)	
	
	if (len(faces)> 0):
		[x,y,w,h] = faces[0]
		
		#for (x,y,w,h) in faces:

		
		face = img[y:y+h, x:x+w] # On crop le visage/

		

		pic = Image.fromarray(face)

		img = pic.resize((28,28))

		#img = Image.open(path, 'r').resize((28,28))

		# Passe l'image en niveau de gris
		img = ImageOps.grayscale(img)

		# img -> data
		imgdata = img.getdata()
		image_tab = numpy.array(imgdata)
		#print(image_tab[0:99])
		X_test[0] = image_tab[0:784]
		X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
	
		Z = model.predict(X_test, batch_size=32, verbose=0)
		print(Z[0])

	else:
		print("error : no face detected")


