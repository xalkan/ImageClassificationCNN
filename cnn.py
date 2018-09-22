# Convolutional Neural Network for Image CLassification

# 1 -  Building the CNN
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

 # initialize the CNN
 classifier = Sequential()

 # adding convolution layer
 # arg 1 - number of feature detectors = feature maps
 # arg 2 and 3 - row and col of feature detector
 # default border mode
 # input_shape - input image format - (256, 256, 3) means colored images of 3 RGB channels with size
 # equal to 256x256 pixels
 # input_shape = (3, 256, 256) for theano backend
 # input_shape = (256, 256, 3) for tensorflow backend
 classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu' ))
