#!/usr/bin/env python

import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1' # cpu -1, gpus 0,1
os.environ['PYTHONHASHSEED'] = '0' # set seed hash


from keras.models import Sequential, Model
from keras.layers import Dense,Activation, Input
from keras.layers.embeddings import Embedding
from keras.optimizers import TFOptimizer

import time
import random
import numpy as np
import tensorflow as tf

architectures = ['/cpu:0','/gpu:0']

np.random.seed(1)
random.seed(1)
tf.set_random_seed(1)

n_class = 1000
n_word = 10000
n = 5000
p = 50

n_epoch = 5
n_batch = 128

t = dict()

x_train=np.random.randint(n_word,size=(n,p))
y_train=np.random.randint(n_class,size=(n,p,1))

inputs = Input(shape=(p,))
embedding_vec = Embedding(n_word,512)(inputs)
d1 = Dense(256, activation='relu')(embedding_vec)
d2 = Dense(128, activation='relu')(d1)
d3 = Dense(64, activation='relu')(d2)
d4 = Dense(n_class, activation='softmax')(d3)
model = Model(inputs=inputs,outputs=d4)

for arch in architectures:

	with tf.device(arch):

		model.compile(loss='sparse_categorical_crossentropy',optimizer='sgd')

		#print(model.summary())

		print('Benchmarking ' + arch + '.')

		t1 = time.time()
		model.fit(x_train,y_train,epochs=n_epoch,batch_size=n_batch,verbose=1)
		t2 = time.time()
		t[arch] = t2-t1

print('\n\n\n\n')
print(architectures[0] + ': ' + str(round((t[architectures[0]])/60,4)) + 'm (' + str(round((t[architectures[0]])/n_epoch,4)) + 's per epoch).\n' + \
	  architectures[1] + ': ' + str(round((t[architectures[1]])/60,4)) + 'm (' + str(round((t[architectures[1]])/n_epoch,4)) + 's per epoch).\n')