#!/usr/bin/env python

import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1' # cpu -1, gpus 0,1
os.environ['PYTHONHASHSEED'] = '0' # set seed hash

from keras.models import Model
from keras.layers import Dense, Activation, Input, Flatten, Dropout
from keras.layers.embeddings import Embedding
from keras.constraints import maxnorm
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.utils import to_categorical
from keras.callbacks import Callback

import time
import random
import numpy as np
import tensorflow as tf
import six.moves.cPickle
from sklearn.model_selection import ParameterGrid

import geo_data as geo
import benchmarking as bm

class Benchmarking(Callback):

    def on_train_begin(self, logs={}):
        self.log = []

    def on_epoch_begin(self, epoch, logs={}):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs={}):
    	self.log.append(time.time() - self.start_time)

architectures = ['/cpu:0','/gpu:0']

n_epoch = 10

model_dict = {'m1':bm.m1,'m2':bm.m2,'m3':bm.m3,'m4':bm.m4,'m5':bm.m5,'m6':bm.m6,'m7':bm.m7}

params_sweep = {'model':['m' + str(n) for n in list(range(1,8))],
                'n':[100,500,1000],
                'n_batch':[32,64,128],
                'n_read':[20,50]}

param_grid = list(ParameterGrid(params_sweep))

params = {'d_emb':128,
          'd_cnn1':32,
          'd_cnn2':32,
          'k1':2,
          'k2':2,
          'd_d1':256,
          'd_d2':128,
          'd_d3':64}

results = []

for grid in param_grid:

	np.random.seed(1)
	random.seed(1)
	tf.set_random_seed(1)

	gen, train_generator, _, _ = geo.data(k=4,seed=1,window=4,n_batch=grid['n'],n_reads=grid['n_read'],n_pad=True,n_batch_val=50,shuffle=True,onehot=False)
	train = next(train_generator)

	x_train = train[0]
	y_train = train[1]

	params['p'] = gen.max_len
	params['n_class'] = gen.n_classes
	params['n_word'] = gen.n_vocab

	t = dict()
	logger = dict()

	model = model_dict[grid['model']](**params)

	try:

		for arch in architectures:

			with tf.device(arch):

				np.random.seed(1)
				random.seed(1)
				tf.set_random_seed(1)

				model.compile(loss='categorical_crossentropy',optimizer='sgd')

				print('Benchmarking ' + arch + '.')

				timer = Benchmarking()

				t1 = time.time()
				model.fit(x_train,y_train,epochs=n_epoch,batch_size=grid['n_batch'],class_weight='auto',callbacks=[timer],verbose=1)
				t2 = time.time()
				t[arch] = t2-t1

				gb, tr_params = bm.get_memory(grid['n_batch'],model)

				logger[arch] = {'time':timer.log,
								'gb':gb,
								'n_params':tr_params,
								'model_params':params,
								'data_params':grid}

		results.append(logger)
		six.moves.cPickle.dump(results,open('D:/Dropbox/cnn_gpu_cpu_benchmarking_tmp.pkl','wb'))

		print('\n\n\n\n')
		print(architectures[0] + ': ' + str(round((t[architectures[0]])/60,4)) + 'm (' + str(round((t[architectures[0]])/n_epoch,4)) + 's per epoch).\n' + \
			  architectures[1] + ': ' + str(round((t[architectures[1]])/60,4)) + 'm (' + str(round((t[architectures[1]])/n_epoch,4)) + 's per epoch).\n')


	except:

		continue

six.moves.cPickle.dump(results,open('D:/Dropbox/cnn_gpu_cpu_benchmarking.pkl','wb'))
