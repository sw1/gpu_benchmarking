#!/usr/bin/env python

from __future__ import print_function

import os
import sys
import six.moves.cPickle
import csv
import numpy as np
import random
from sklearn.model_selection import train_test_split
from itertools import product
from collections import Counter
from keras.utils import to_categorical

class GenerateBatch(object):

    def __init__(self, seed, read_len, window, n_vocab, n_batch = 32, n_classes = 2, n_reads = 250, n_pad = 21, 
            shuffle = True, onehot = False):
        'Initialization'
        self.read_len = read_len
        self.window = window
        self.n_vocab = n_vocab
        self.n_classes = n_classes
        self.n_batch = n_batch
        self.n_reads = n_reads
        self.n_pad = n_pad
        self.max_len = n_reads * (read_len + n_pad)
        self.shuffle = shuffle
        self.onehot = onehot
        self.seed = seed

        if onehot:
            self.in_len = (self.max_len, self.n_vocab) 
            self.in_type = 'float32'
        else:
            self.in_len = (self.max_len,) 
            self.in_type = 'int32'

    def __get_exploration_order(self, ids):
        'Generates order of exploration'
        # Find exploration order
        if self.shuffle == True:
            random.seed(self.seed)
            random.shuffle(ids)

        return ids

    def __data_generation(self, fns, labels, ids, nb):
        'Generates data of batch_size samples'

        # Initialization
        batch_X = np.zeros((nb, self.max_len), dtype=int)
        batch_y = np.zeros((nb, self.n_classes), dtype=np.float32)

        for i,id in enumerate(ids):

            x = six.moves.cPickle.load(open(fns[id],'rb'))
            batch_y[i][labels[id]] = 1.0
            read_idxs = random.sample(range(len(x)),self.n_reads)

            for r,idx in enumerate(read_idxs):
                batch_X[i,r*self.read_len + r*self.n_pad:(r+1)*self.read_len + r*self.n_pad] = x[idx]

        if self.onehot:
            batch_X = np.apply_along_axis(to_categorical,1,batch_X,num_classes=self.n_vocab)

        return batch_X, batch_y

    def generate(self, fns, labels, ids, nb=0):
        'Generates batches of samples'
        
        if nb == 0:
            nb = self.n_batch

        # Infinite loop
        while True:
            # Generate order of exploration of dataset
            ids_tmp = self.__get_exploration_order(ids)

            random.seed(self.seed)

            # Generate batches
            end = int(len(ids)/nb)
            for i in range(end):
                # Find list of IDs
                batch_ids = ids_tmp[i*nb:(i+1)*nb]
                # Generate data
                batch_X, batch_y = self.__data_generation(fns, labels, batch_ids, nb)

                yield batch_X, batch_y

    
def data(k,seed,window,n_batch,n_reads,n_batch_val=0,n_pad=False,shuffle=True,onehot=False):

    params = {'window': window,
              'n_batch': n_batch,  
              'n_reads': n_reads,
              'onehot': onehot,
              'shuffle': shuffle,
              'seed': seed}
    
    if n_pad:
        params['n_pad'] = window * k
    else:
        params['n_pad'] = 0
    
    nts = ['A','C','G','T']
    key = {''.join(kmer):i+1 for i,kmer in enumerate(product(nts,repeat=k))}
    key['PAD'] = 0
    
    params['n_vocab'] = len(key)

    kmer_dir = 'data/kmers_' + str(k)

    meta_fn = 'data/labels.csv'

    fns = {f.split('.pkl')[0]:os.path.join(kmer_dir, f) for f in os.listdir(kmer_dir)
                       if os.path.isfile(os.path.join(kmer_dir, f))}

    meta = csv.reader(open(meta_fn,'r'), delimiter=',', quotechar='"')
    meta = [(r,l) for r,l in meta if r in fns]

    train, test = train_test_split(meta,test_size=0.2,random_state=seed,
            shuffle=True,stratify=[l for r,l in meta])
    val, test = train_test_split(test,test_size=0.5,random_state=seed,
            shuffle=True,stratify=[l for r,l in test])

    ids_train = [r for r,l in train]
    ids_val = [r for r,l in val]
    ids_test = [r for r,l in test]

    ids = {'train':ids_train,'val':ids_val,'test':ids_test}
    labels = {r:int(l) for r,l in meta}

    read_len = len(six.moves.cPickle.load(open(next(iter(fns.values())),'rb'))[0])
    params['read_len'] = read_len
    
    n_classes = max(labels.values()) + 1
    params['n_classes'] = n_classes

    gen = GenerateBatch(**params)
    
    train_generator = gen.generate(fns,labels,ids['train'])
    val_generator = gen.generate(fns,labels,ids['val'],nb=n_batch_val)

    return gen, train_generator, val_generator, ids