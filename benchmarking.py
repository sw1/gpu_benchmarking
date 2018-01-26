import csv
import numpy as np

from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Activation, Input, Flatten, Dropout
from keras.layers.embeddings import Embedding
from keras.constraints import maxnorm
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.utils import to_categorical
from keras.callbacks import Callback

def dump_results(results,path='results.csv'):
    
    architectures = ['/gpu:0','/cpu:0']

    with open(path, 'w', newline='') as csvfile:

        f = csv.writer(csvfile, delimiter=',')

        headers = []
        headers.append('arch')
        headers.extend([h[2:] for h in list(results[0][architectures[0]]['data_params'].keys())[0:3]])
        headers.extend(list(results[0][architectures[0]]['data_params'].keys())[3])
        headers.extend(list(results[0][architectures[0]]['model_params'].keys()))
        headers.append('n_params')
        headers.append('gb')
        headers.extend(list(results[0][architectures[0]]['time'][0].keys()))

        f.writerow(headers)

        for result in results:
            if set(architectures) == set(result.keys()):
                for arch in architectures:
                    values = []
                    values.append(arch)
                    values.extend(list(result[arch]['data_params'].values()))
                    values.extend(list(result[arch]['model_params'].values()))
                    values.append(result[arch]['n_params'])
                    values.append(result[arch]['gb'])

                    time = [list(d.values()) for d in result[arch]['time']]

                    for t in time:
                        f.writerow(values + t)

def get_memory(n_batch, model, gb = True):

    counter = 0
    for layer in model.layers:
        layer_memory = 1
        for s in layer.output_shape:
            if s is None:
                continue
            layer_memory *= s
        counter += layer_memory

    tr_params= np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    nontr_params = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    memory_per_batch = 4.0*n_batch*(counter + tr_params + nontr_params)

    if gb:
        memory_per_batch = memory_per_batch/(1024.0 ** 3)

    return memory_per_batch, tr_params

def m1(n_class,p,n_word,d_emb2,d_d2,**kwargs):

    inputs = Input(shape=(p,))
    layer_embed = Embedding(n_word,d_emb2)(inputs)
    layer_flatten = Flatten()(layer_embed)
    layer_dense = Dense(d_d2, activation='relu', kernel_constraint=maxnorm(3))(layer_flatten)
    layer_dropout = Dropout(.5)(layer_dense)
    outputs = Dense(n_class, activation='softmax')(layer_dropout)
    model = Model(inputs=inputs,outputs=outputs)

    return(model)

def m2(n_class,p,n_word,d_emb2,d_d2,d_d3,**kwargs):

    inputs = Input(shape=(p,))
    layer_embed = Embedding(n_word,d_emb2)(inputs)
    layer_flatten = Flatten()(layer_embed)
    layer_dense_1 = Dense(d_d2, activation='relu', kernel_constraint=maxnorm(3))(layer_flatten)
    layer_dense_2 = Dense(d_d3, activation='relu', kernel_constraint=maxnorm(3))(layer_dense_1)
    layer_dropout = Dropout(.5)(layer_dense_2)
    outputs = Dense(n_class, activation='softmax')(layer_dropout)
    model = Model(inputs=inputs,outputs=outputs)

    return(model)

def m3(n_class,p,n_word,d_emb2,d_d2,d_d3,d_d4,**kwargs):

    inputs = Input(shape=(p,))
    layer_embed = Embedding(n_word,d_emb2)(inputs)
    layer_flatten = Flatten()(layer_embed)
    layer_dense_1 = Dense(d_d2, activation='relu', kernel_constraint=maxnorm(3))(layer_flatten)
    layer_dense_2 = Dense(d_d3, activation='relu', kernel_constraint=maxnorm(3))(layer_dense_1)
    layer_dense_3 = Dense(d_d4, activation='relu', kernel_constraint=maxnorm(3))(layer_dense_2)
    layer_dropout = Dropout(.5)(layer_dense_3)
    outputs = Dense(n_class, activation='softmax')(layer_dropout)
    model = Model(inputs=inputs,outputs=outputs)

    return(model)

def m4(n_class,p,n_word,d_emb1,d_cnn1,k1,d_d1,**kwargs):

    inputs = Input(shape=(p,))
    layer_embed = Embedding(n_word,d_emb1)(inputs)
    layer_cnn = Conv1D(d_cnn1, 
                       kernel_size=k1, padding='same', activation='relu',
                       kernel_constraint=maxnorm(3))(layer_embed)
    layer_flatten = Flatten()(layer_cnn)
    layer_dense = Dense(d_d1, activation='relu', kernel_constraint=maxnorm(3))(layer_flatten)
    layer_dropout = Dropout(.5)(layer_dense)
    outputs = Dense(n_class, activation='softmax')(layer_dropout)
    model = Model(inputs=inputs,outputs=outputs)

    return(model)

def m5(n_class,p,n_word,d_emb1,d_cnn1,k1,d_d1,d_d2,**kwargs):

    inputs = Input(shape=(p,))
    layer_embed = Embedding(n_word,d_emb1)(inputs)
    layer_cnn = Conv1D(d_cnn1, 
                       kernel_size=k1, padding='same', activation='relu',
                       kernel_constraint=maxnorm(3))(layer_embed)
    layer_flatten = Flatten()(layer_cnn)
    layer_dense_1 = Dense(d_d1, activation='relu', kernel_constraint=maxnorm(3))(layer_flatten)
    layer_dense_2 = Dense(d_d2, activation='relu', kernel_constraint=maxnorm(3))(layer_dense_1)
    layer_dropout = Dropout(.5)(layer_dense_2)
    outputs = Dense(n_class, activation='softmax')(layer_dropout)
    model = Model(inputs=inputs,outputs=outputs)

    return(model)

def m6(n_class,p,n_word,d_emb1,d_cnn1,k1,d_d1,d_d2,d_d3,**kwargs):

    inputs = Input(shape=(p,))
    layer_embed = Embedding(n_word,d_emb1)(inputs)
    layer_cnn = Conv1D(d_cnn1, 
                       kernel_size=k1, padding='same', activation='relu',
                       kernel_constraint=maxnorm(3))(layer_embed)
    layer_flatten = Flatten()(layer_cnn)
    layer_dense_1 = Dense(d_d1, activation='relu', kernel_constraint=maxnorm(3))(layer_flatten)
    layer_dense_2 = Dense(d_d2, activation='relu', kernel_constraint=maxnorm(3))(layer_dense_1)
    layer_dense_3 = Dense(d_d3, activation='relu', kernel_constraint=maxnorm(3))(layer_dense_2)
    layer_dropout = Dropout(.5)(layer_dense_3)
    outputs = Dense(n_class, activation='softmax')(layer_dropout)
    model = Model(inputs=inputs,outputs=outputs)

    return(model)

def m7(n_class,p,n_word,d_emb1,d_cnn1,d_cnn2,k1,k2,d_d1,d_d2,d_d3,**kwargs):

    inputs = Input(shape=(p,))
    layer_embed = Embedding(n_word,d_emb1)(inputs)
    layer_cnn_1 = Conv1D(d_cnn1, 
                       kernel_size=k1, padding='same', activation='relu',
                       kernel_constraint=maxnorm(3))(layer_embed)
    layer_dropout_1 = Dropout(.1)(layer_cnn_1)
    layer_cnn_2 = Conv1D(d_cnn2, 
                         kernel_size=k2, padding='same', activation='relu',
                         kernel_constraint=maxnorm(3))(layer_dropout_1)
    layer_flatten = Flatten()(layer_cnn_2)
    layer_dense_1 = Dense(d_d1, activation='relu', kernel_constraint=maxnorm(3))(layer_flatten)
    layer_dense_2 = Dense(d_d2, activation='relu', kernel_constraint=maxnorm(3))(layer_dense_1)
    layer_dense_3 = Dense(d_d3, activation='relu', kernel_constraint=maxnorm(3))(layer_dense_2)
    layer_dropout_2 = Dropout(.5)(layer_dense_3)
    outputs = Dense(n_class, activation='softmax')(layer_dropout_2)
    model = Model(inputs=inputs,outputs=outputs)

    return(model)