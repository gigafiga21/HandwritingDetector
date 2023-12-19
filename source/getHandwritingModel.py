import os
import warnings

import numpy as np

from tensorflow.python.util import deprecation
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.datasets import mnist


deprecation._PRINT_DEPRECATION_WARNINGS = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

def generateHandwritingModel():
	(X_train, y_train), (X_test, y_test) = mnist.load_data()
	X_train = X_train / 255
	X_test = X_test / 255
	y_train = to_categorical(y_train, 10)
	y_test = to_categorical(y_test, 10)
	X_train, X_test = X_train.reshape((60000, 28, 28, 1)), X_test.reshape((10000, 28, 28, 1))
	input_size = X_train[0].shape

	cnn = Sequential()
	cnn.add(Conv2D(64, (3, 3), input_shape = input_size, activation = 'selu', padding = 'same'))
	cnn.add(Dropout(0.25))
	cnn.add(Conv2D(32, (3, 3), activation = 'selu', padding = 'same'))
	cnn.add(MaxPooling2D(pool_size = (2, 2)))
	cnn.add(Dropout(0.5))
	cnn.add(Flatten())
	cnn.add(Dense(10, activation = 'softmax'))
	cnn.compile(loss = 'categorical_crossentropy', optimizer = 'nadam', metrics = ['accuracy'])

	cnn.fit(X_train, y_train, epochs = 3, batch_size = 64, validation_data = (X_test, y_test))

	cnn.save('handwritingModel.h5')
	return cnn

def loadHandwritingModel():
	return load_model('handwritingModel.h5')

def getHandwritingModel():
	try:
		return loadHandwritingModel()
	except:
		print('generating model...')
		return generateHandwritingModel()
