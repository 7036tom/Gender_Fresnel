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



# Create the haar cascade
cascPath = 'haarcascade_frontalface_default.xml'
profile_CascPath = 'haarcascade_profileface.xml'
faceCascade = cv2.CascadeClassifier(cascPath)
profile_faceCascade = cv2.CascadeClassifier(profile_CascPath)

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load dataset
dataframe = pandas.read_csv("Database1010feret", header=None, sep=" ")
dataset = dataframe.values

# Not wholly used
X_test = dataset[2580:2600,0:100]


# Load pretrained model. Details can be found in the model file
model = load_model('model_naif')

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
 	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	# Detect faces in the image
	faces = faceCascade.detectMultiScale(frame, 1.1,  5)
	
	
	for (x,y,w,h) in faces:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
		roi_gray = gray[y:y+h, x:x+w]
		
		roi_gray = imutils.resize(roi_gray, width=10, height=10)

		img_tab = range(0,100)
		for i in range(0,10):
			for j in range(0,9):	
				img_tab[i+j*10]= roi_gray[j][i]
		X_test[0] = img_tab[0:100]
		
	
		Z = model.predict(X_test, batch_size=32, verbose=0)
		if (Z[0][0]*2 < Z[0][1]):
			Genre = "homme"
		elif (Z[0][0] > 2*Z[0][1]):
			Genre = "femme"
		else:
			Genre = "?"
		
		cv2.putText(frame,Genre,(x,y), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
 	

	# Detect profiles in the image

	#profiles = profile_faceCascade.detectMultiScale(frame, 1.1,  5)

	#for (x,y,w,h) in profiles:
	#	cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
	#	roi_gray = gray[y:y+h, x:x+w]
	#	
	#	roi_gray = imutils.resize(roi_gray, width=10, height=10)

	#	img_tab = range(0,99)
	#	for i in range(0,9):
	#		for j in range(0,9):	
	#			img_tab[i+j*10]= roi_gray[j][i]
	#	X_test[0] = img_tab[0:99]
	
	#	Z = model.predict(X_test, batch_size=32, verbose=0)
	#	if (Z[0][0]*2 < Z[0][1]):
	#		Genre = "homme"
	#	elif (Z[0][0] > 2*Z[0][1]):
	#		Genre = "femme"
	#	else:
	#		Genre = "?"
		
	#	cv2.putText(frame,Genre,(x,y), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)

	cv2.imshow("cam",frame)
	key = cv2.waitKey(1) & 0xFF
	#if the `q` key is pressed, break from the lop
	if key == ord("q"):
		break
# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()		
	

	

	

	

