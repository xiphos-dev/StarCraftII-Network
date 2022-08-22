#!/usr/bin/env python
# coding: utf-8

import pickle
import os
import numpy as np



with open("X_train_unico","rb") as arc:
    X_train = pickle.load(arc)

with open("X_test_unico","rb") as arc:
    X_test = pickle.load(arc)

with open("y_train_unico","rb") as arc:
    y_train = pickle.load(arc)

with open("y_test_unico","rb") as arc:
    y_test = pickle.load(arc)

with open("vocabulario", "rb") as arc:
    vocabulario = pickle.load(arc)

with open("encoder_mid", "rb") as arc:
    encoder_mid = pickle.load(arc)


input_shape_rnn = X_train.shape[1]

categorias = y_train.shape[1]


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

batch_size = 128

units = 64
output_size = 10 

lr = 5e-3


def modeloRNN(num_unidades,vocab,embedding_dim=256,categorias=2):
    model = keras.Sequential()
    model.add(layers.Embedding(vocab, embedding_dim, mask_zero=True, input_length=input_shape_rnn))

    #model.add(layers.GRU(32))
    
    model.add(layers.LSTM(num_unidades))
    #model.add(layers.SimpleRNN(10))

    #model.add(layers.Dense(128, activation='relu'))
    #model.add(layers.Dropout(.2))

    model.add(layers.Dense(categorias, activation="softmax"))

    model.summary()
    return model

model = modeloRNN(X_train.shape[1],len(vocabulario)+1,256,len(encoder_mid.values()))


ruta_modelo = "../../../Modelo_recurrente/Zerg/Input_unico_z/"
checkpoint_best = ModelCheckpoint(filepath=ruta_modelo, monitor='val_accuracy',verbose=1, save_best_only=True, mode='auto')
lrschedule_1 = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.70, mode='auto')

model.compile(
    loss=keras.losses.CategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(lr=lr), 
    metrics=["accuracy"]
)

historia = model.fit(
    X_train_rnn, y_train, 
    validation_data=(X_test_rnn, y_test), 
    batch_size=batch_size, 
    epochs=10,
    callbacks=[checkpoint_best, lrschedule_1]
)



with open('../../../Modelo_recurrente/Zerg/historia_input_unico_z', 'wb') as file_pi:
    pickle.dump(historia.history, file_pi)
