#!/usr/bin/env python

import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1' # cpu -1, gpus 0,1
os.environ['PYTHONHASHSEED'] = '0' # set seed hash


from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, Flatten, Dropout
from keras.layers.embeddings import Embedding
from keras.constraints import maxnorm
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.utils import to_categorical

import time
import random
import numpy as np
import tensorflow as tf

architectures = ['/cpu:0','/gpu:0']

np.random.seed(1)
random.seed(1)
tf.set_random_seed(1)

n_class = 3 # 1000
n_word = 256 # 10000
n = 500 # 1200 # 5000
p = 10000 # 50

n_epoch = 10
n_batch = 64

t = dict()

x_train = np.random.randint(n_word,size=(n,p))
y_train = np.random.randint(n_class,size=(n))
y_train = np.apply_along_axis(to_categorical,0,y_train,num_classes=n_class)

inputs = Input(shape=(p,))
layer_embed = Embedding(n_word,128)(inputs)
layer_cnn_1 = Conv1D(32, 
                   kernel_size=2, padding='same', activation='relu',
                   kernel_constraint=maxnorm(3))(layer_embed)
layer_dropout_1 = Dropout(.1)(layer_cnn_1)
layer_cnn_2 = Conv1D(32, 
                     kernel_size=3, padding='same', activation='relu',
                     kernel_constraint=maxnorm(3))(layer_dropout_1)
# layer_pooling = MaxPooling1D(pool_size=2)(layer_cnn_2)
layer_flatten = Flatten()(layer_cnn_2)
layer_dense_1 = Dense(256, activation='relu', kernel_constraint=maxnorm(3))(layer_flatten)
layer_dense_2 = Dense(128, activation='relu', kernel_constraint=maxnorm(3))(layer_dense_1)
layer_dense_3 = Dense(64, activation='relu', kernel_constraint=maxnorm(3))(layer_dense_2)
layer_dropout_2 = Dropout(.5)(layer_dense_3)
outputs = Dense(n_class, activation='softmax')(layer_dropout_2)
model = Model(inputs=inputs,outputs=outputs)

for arch in architectures:

	with tf.device(arch):

		model.compile(loss='categorical_crossentropy',optimizer='sgd')

		#print(model.summary())

		print('Benchmarking ' + arch + '.')

		t1 = time.time()
		model.fit(x_train,y_train,epochs=n_epoch,batch_size=n_batch,verbose=1)
		t2 = time.time()
		t[arch] = t2-t1

print('\n\n\n\n')
print(architectures[0] + ': ' + str(round((t[architectures[0]])/60,4)) + 'm (' + str(round((t[architectures[0]])/n_epoch,4)) + 's per epoch).\n' + \
	  architectures[1] + ': ' + str(round((t[architectures[1]])/60,4)) + 'm (' + str(round((t[architectures[1]])/n_epoch,4)) + 's per epoch).\n')