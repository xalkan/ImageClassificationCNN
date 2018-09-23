# Convolutional Neural Network for Image CLassification

# 1 -  Building the CNN
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# initialize the CNN
classifier = Sequential()

# adding convolutional layer
# arg 1 - number of feature detectors = feature maps
# arg 2 and 3 - row and col of feature detector
# default border mode
# input_shape - input image format - (256, 256, 3) means colored images of 3 RGB channels with size
# equal to 256x256 pixels
# input_shape = (3, 256, 256) for theano backend
# input_shape = (256, 256, 3) for tensorflow backend
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu' ))

# adding pooling layer
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# adding another convolutional layer with max pooling
classifier.add(Conv2D(32, (3, 3), activation = 'relu' ))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# flattening the input image
classifier.add(Flatten())

# add classic nn
# add fully connected layers
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# compiling the nn
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# 2 - Fitting the cnn to the image classifier after image augmentation
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set', target_size=(64, 64), batch_size=32, class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set', target_size=(64, 64), batch_size=32, class_mode='binary')

classifier.fit_generator(training_set, steps_per_epoch=8000, epochs=1, validation_data=test_set, validation_steps=2000)

