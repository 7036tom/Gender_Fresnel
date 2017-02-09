# coding: utf-8

import PIL
import Image
import numpy
import ImageOps
import sys
import os

numpy.set_printoptions(threshold=numpy.nan)

Dim = 28


for path in os.listdir('/home/tom/Documents/Projets python/Fresnel/Genre_Deep_Learning/feret/Test/female'):
	# On ouvre l'image et on la redimensionne
	#path = sys.argv[1];
	img = Image.open('Test/female/'+path, 'r').resize((Dim,Dim))
	
	# Passe l'image en niveau de gris
	img = ImageOps.grayscale(img)
	
	# img -> data
	imgdata = img.getdata()
	image_tab = numpy.array(imgdata)


	# Mettons a present la liste 1D sous forme de matrice
	larg, haut = img.size
	matrix_image= numpy.reshape(image_tab,(haut,larg))

	for i in range(0,Dim*Dim):
		print(image_tab[i])
	print(0)

for path in os.listdir('/home/tom/Documents/Projets python/Fresnel/Genre_Deep_Learning/feret/Test/male'):
	# On ouvre l'image et on la redimensionne
	#path = sys.argv[1];
	img = Image.open('Test/male/'+path, 'r').resize((Dim,Dim))
	
	# Passe l'image en niveau de gris
	img = ImageOps.grayscale(img)
	
	# img -> data
	imgdata = img.getdata()
	image_tab = numpy.array(imgdata)


	# Mettons a present la liste 1D sous forme de matrice
	larg, haut = img.size
	matrix_image= numpy.reshape(image_tab,(haut,larg))

	for i in range(0,Dim*Dim):
		print(image_tab[i])
	print(1)
