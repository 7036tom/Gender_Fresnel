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


init = dataset[2595:2600,0:784]

model40 = load_model('model_Convo4.0')
model30 = load_model('model_Convo3.0')
model20 = load_model('model_Convo2.0')

usr_input = 0

while (usr_input != 1):
	X_test = dataset[2580:2600,0:784]
	pathDoc = raw_input("Veuillez entrer le chemin du dossier que vous voulez trier")
	pathDoc = pathDoc[1:len(pathDoc)-2]
	
	print(pathDoc)
	for imgpath in os.listdir(pathDoc):
		X_test20 = init
		X_test30 = init
		X_test40 = init
		
		path = pathDoc+"/"+imgpath 
		print(path)
		
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

			X_test20[0] = image_tab[0:784]
			X_test30[0] = image_tab[0:784]
			X_test40[0] = image_tab[0:784]
			
			X_test20 = X_test20.reshape(X_test20.shape[0], 1, 28, 28).astype('float32')
			X_test30 = X_test30.reshape(X_test30.shape[0], 1, 28, 28).astype('float32')
			X_test40 = X_test40.reshape(X_test40.shape[0], 1, 28, 28).astype('float32')
	
			Z20 = model20.predict(X_test20, batch_size=32, verbose=0)
			Z30 = model30.predict(X_test30, batch_size=32, verbose=0)
			Z40 = model40.predict(X_test40, batch_size=32, verbose=0)
			
			
			if ((Z20[0][0]+Z30[0][0])< Z20[0][1]+Z30[0][1]):
				pic.save('/media/tom/World/wikitrié/male/'+imgpath)
				print("male")
			elif (Z20[0][0]+Z30[0][0]> (Z20[0][1]+Z30[0][1])):
				pic.save('/media/tom/World/wikitrié/female/'+imgpath)
				print("female")
			"""

			if ((Z40[0][0])< Z40[0][1]):
				pic.save('/media/tom/World/wikitrié/male/'+imgpath)
				print("male")
			elif (Z40[0][0] > Z40[0][1]):
				pic.save('/media/tom/World/wikitrié/female/'+imgpath)
				print("female")
			"""
		else:
			print("error : no face detected")


