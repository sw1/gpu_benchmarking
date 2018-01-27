#!/usr/bin/env python

import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1' # cpu -1, gpus 0,1
os.environ['PYTHONHASHSEED'] = '0' # set seed hash

from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Activation, Input, Flatten, Dropout
from keras.layers.embeddings import Embedding
from keras.constraints import maxnorm
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.utils import to_categorical
from keras.callbacks import Callback

import gc
import sys
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
    	self.end_time = time.time()
    	log_dict = {'start':self.start_time,
    			    'end':self.end_time,
    			    'elapsed':self.end_time - self.start_time,
    			    'epoch':epoch}
    	self.log.append(log_dict)

architectures = ['/gpu:0','/cpu:0']

n_epoch = 20

model_dict = {'m1':bm.m1,'m2':bm.m2,'m3':bm.m3,'m4':bm.m4,'m5':bm.m5,'m6':bm.m6,'m7':bm.m7}

params_sweep = {'n':[500],
                'c_model':['m' + str(n) for n in list(range(1,8))],
                'b_n_batch':[128,96,64,32],
                'a_n_read':[40,30,20,10]}

param_grid = ParameterGrid(params_sweep)

params = {'d_emb1':128, # 128
          'd_cnn1':32,
          'd_cnn2':32,
          'k1':3, # 2
          'k2':3, # 2
          'd_d1':128, # 256
          'd_d2':64, # 128
          'd_d3':32} # 32

failures = []
results = []
n_reads_tmp = 0

stdout_tmp = sys.stdout
sys.stdout = open('D:/Active Research/cnn_lstm/results/model.out', 'w')

for grid in param_grid:

	np.random.seed(1)
	random.seed(1)
	tf.set_random_seed(1)

	print('Generating ' + str(grid['n']) + ' batches of ' + str(grid['a_n_read']) + ' reads each.')

	if n_reads_tmp != grid['a_n_read']:

		gen, train_generator, _, _ = geo.data(k=4,seed=1,window=4,n_batch=grid['n'],n_reads=grid['a_n_read'],n_pad=True,n_batch_val=50,shuffle=True,onehot=False)
		train = next(train_generator)

		x_train = train[0]
		y_train = train[1]

		n_reads_tmp = grid['a_n_read']

	params['p'] = gen.max_len
	params['n_class'] = gen.n_classes
	params['n_word'] = gen.n_vocab

	t = dict()
	logger = dict()

	try:

		for arch in architectures:

			print('Benchmarking ' + arch + '.')
			print('Param grid: ')
			print(grid)

			with tf.device(arch):

				np.random.seed(1)
				random.seed(1)
				tf.set_random_seed(1)

				model = model_dict[grid['c_model']](**params)
				model.compile(loss='categorical_crossentropy',optimizer='sgd')

				print(model.summary())

				timer = Benchmarking()

				t1 = time.time()
				model.fit(x_train,y_train,epochs=n_epoch,batch_size=grid['b_n_batch'],class_weight='auto',callbacks=[timer],verbose=1)
				t2 = time.time()
				t[arch] = t2-t1

				gb, tr_params = bm.get_memory(grid['b_n_batch'],model)

				logger[arch] = {'time':timer.log,
								'gb':gb,
								'n_params':tr_params}

				# garbage collection
				del model
				K.clear_session()
				gc.collect()
								

		results.append({'status':1,'log':logger,'model_params':params,'data_params':grid})
		six.moves.cPickle.dump(results,open('D:/Dropbox/cnn_gpu_cpu_benchmarking_tmp.pkl','wb'))

		print(architectures[0] + ': ' + str(round((t[architectures[0]])/60,4)) + 'm (' + str(round((t[architectures[0]])/n_epoch,4)) + 's per epoch).\n' + \
			  architectures[1] + ': ' + str(round((t[architectures[1]])/60,4)) + 'm (' + str(round((t[architectures[1]])/n_epoch,4)) + 's per epoch).\n')


	except:

		print('Likely memory error; Skipping')
		print('Param grid: ')
		print(grid)

		results.append({'status':0,'data_params':grid})

		continue


print('Failures: ')
[print(x['data_params']) for x in results if x['status'] == 0]

print('Dumping results.')
six.moves.cPickle.dump(results,open('D:/Dropbox/cnn_gpu_cpu_benchmarking.pkl','wb'))
six.moves.cPickle.dump(results,open('D:/Active Research/cnn_lstm/results/cnn_gpu_cpu_benchmarking.pkl','wb'))
bm.dump_results(results,path='D:/Active Research/cnn_lstm/results/results.csv')
print('Complete.')

sys.stdout = stdout_tmp