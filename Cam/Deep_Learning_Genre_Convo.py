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
from keras.models import load_model
import numpy
import pandas

from keras import backend as K
K.set_image_dim_ordering('th') # Avoid input to Flatten is not fully defined


# Create the haar cascade
cascPath = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load dataset
dataframe = pandas.read_csv("database2828", header=None, sep=" ")
dataset = dataframe.values


# Load pretrained model. Details can be found in the model file
#model = load_model('model_Convo')
model30 = load_model('model_Convo3.0')
model20 = load_model('model_Convo2.0')

#Cam input

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())
 
# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
	camera = cv2.VideoCapture(0)
	time.sleep(0.25)
 
# otherwise, we are reading from a video file
else:
	camera = cv2.VideoCapture(args["video"])
 
# initialize the first frame in the video stream
firstFrame = None
# loop over the frames of the video
while True:
	# grab the current frame and initialize the occupied/unoccupied
	# text
	(grabbed, frame) = camera.read()
	text = "Unoccupied"
 
	# if the frame could not be grabbed, then we have reached the end
	# of the video
	if not grabbed:
		break

	frame = imutils.resize(frame, width=500)
	#frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # On passe de BGR a RGB !
 	#gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
	
	# Detect faces in the image
	faces = faceCascade.detectMultiScale(frame, 1.1,  5)
	
	# Not wholly used
	X2_test = dataset[2598:2600,0:784]

	
	
	
	for (x,y,w,h) in faces:

		X_test20 = X2_test
		X_test30 = X2_test
		
		
		
		face = frame[y:y+h, x:x+w] # On crop le visage/

		face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB) # On passe de BGR a RGB !
		pic = Image.fromarray(face)
		img = pic.resize((28,28))
		
		"""
		img.save("test.jpg")
		break
		"""
		
		# Passe l'image en niveau de gris
		img = ImageOps.grayscale(img)

		# img -> data
		imgdata = img.getdata()
		image_tab = numpy.array(imgdata)
		#print(image_tab[0:99])
		

		"""	
		numrows = len(image_tab)    
		numcols = len(image_tab[0]) 

		print(numrows, numcols)

		if (numcols == 28 and numrows == 28):
		"""

		X_test20[0] = image_tab[0:784]
		X_test30[0] = image_tab[0:784]

		X_test20 = X_test20.reshape(X_test20.shape[0], 1, 28, 28).astype('float32')
		X_test30 = X_test30.reshape(X_test30.shape[0], 1, 28, 28).astype('float32')
		
		Z20 = model20.predict(X_test20, batch_size=32, verbose=0)
		Z30 = model30.predict(X_test30, batch_size=32, verbose=0)
		
		if ((Z20[0][0]+Z30[0][0])< Z20[0][1]+Z30[0][1]):
			Genre = "homme"
		elif (Z20[0][0]+Z30[0][0]> (Z20[0][1]+Z30[0][1])):
			Genre = "femme"
		else:
			Genre = "?"

		"""
		if (Z[0][0]*2 < Z[0][1]):
			Genre = "homme"
		elif (Z[0][0] > 2*Z[0][1]):
			Genre = "femme"
		else:
			Genre = "?"
		"""
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
		cv2.putText(frame,Genre,(x,y), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)

	cv2.imshow("cam",frame)
	key = cv2.waitKey(1) & 0xFF
	#if the `q` key is pressed, break from the lop
	if key == ord("q"):
		break
# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()		
	

	

	

	

