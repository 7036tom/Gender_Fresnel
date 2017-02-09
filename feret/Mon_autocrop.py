# import the necessary packages for photo handling

import PIL
import Image
import numpy
import ImageOps
import sys
import os
import imutils
import cv2

# Create the haar cascade
cascPath = 'haarcascade_frontalface_default.xml'

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

################################################## FEMALE ###############################################################"

for path in os.listdir('/home/tom/Documents/Projets python/Fresnel/Genre_Deep_Learning/feret/wiki/femaleOri'):
	imagePath = '/home/tom/Documents/Projets python/Fresnel/Genre_Deep_Learning/feret/wiki/femaleOri/'+path 
	
	# Read the image
	img = cv2.imread(imagePath)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # On passe de BGR a RGB !



	# Detect faces in the image
	faces = faceCascade.detectMultiScale(img, 1.1,  5)	


	for (x,y,w,h) in faces:
		face = img[y:y+h, x:x+w] # On crop le visage/
	
		pic = Image.fromarray(face) # On converti le numpy array en image
		pic.save('/home/tom/Documents/Projets python/Fresnel/Genre_Deep_Learning/feret/cropped_Faces/female/'+path) #On sauvegarde.

################################################## MALE ###############################################################"""

for path in os.listdir('/home/tom/Documents/Projets python/Fresnel/Genre_Deep_Learning/feret/wiki/maleOri'):
	imagePath = '/home/tom/Documents/Projets python/Fresnel/Genre_Deep_Learning/feret/wiki/maleOri/'+path 
	
	# Read the image
	img = cv2.imread(imagePath)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)



	# Detect faces in the image
	faces = faceCascade.detectMultiScale(img, 1.1,  5)	


	for (x,y,w,h) in faces:
		face = img[y:y+h, x:x+w]
	
		pic = Image.fromarray(face)
		pic.save('/home/tom/Documents/Projets python/Fresnel/Genre_Deep_Learning/feret/cropped_Faces/male/'+path)
	
	
