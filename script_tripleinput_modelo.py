#!/usr/bin/env python
# coding: utf-8

import pickle
import os
import numpy as np




with open("./data_rnn_tripleinput/X_train_estructuras","rb") as arc:
    X_train_estructuras = pickle.load(arc)

with open("./data_rnn_tripleinput/X_test_estructuras","rb") as arc:
    X_test_estructuras = pickle.load(arc)

with open("./data_rnn_tripleinput/X_train_unidades","rb") as arc:
    X_train_unidades = pickle.load(arc)

with open("./data_rnn_tripleinput/X_test_unidades","rb") as arc:
    X_test_unidades = pickle.load(arc)

with open("./data_rnn_tripleinput/X_train_mejoras","rb") as arc:
    X_train_mejoras = pickle.load(arc)

with open("./data_rnn_tripleinput/X_test_mejoras","rb") as arc:
    X_test_mejoras = pickle.load(arc)


with open("./data_rnn_tripleinput/y_train","rb") as arc:
    y_train = pickle.load(arc)

with open("./data_rnn_tripleinput/y_test","rb") as arc:
    y_test = pickle.load(arc)


with open("./data_rnn_tripleinput/vocabulario","rb") as arc:
    vocabulario_mid = pickle.load(arc)

with open("./data_rnn_tripleinput/vocabulario_unidades","rb") as arc:
    vocabulario_unidades_mid = pickle.load(arc)

with open("./data_rnn_tripleinput/vocabulario_mejoras","rb") as arc:
    vocabulario_mejoras_mid = pickle.load(arc)

input_shape_rnn = X_train_estructuras.shape[1]
input_shape_mejoras = X_train_mejoras.shape[1]
input_shape_rnn_unidades = X_train_unidades.shape[1]

categorias = y_train.shape[1]



# In[ ]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint


batch_size = 128

units = 64
output_size = 10 

lr = 5e-3


def modeloRNNPuro(input_rnn,input_mejoras,input_rnn_unidades,vocab,vocab_unidades,vocab_mejoras,embedding_dim=256,embedding_unidades=256,embedding_mejoras=256,categorias=2):
    inputs_rnn = keras.Input(shape=(input_rnn,))
    rnn = layers.Embedding(vocab, embedding_dim, input_length=input_rnn)(inputs_rnn)
    rnn = layers.LSTM(input_rnn) (rnn)
    
    inputs_rnn_unidades = keras.Input(shape=(input_rnn_unidades,))
    rnn_unidades = layers.Embedding(vocab_unidades, embedding_unidades, input_length=input_rnn_unidades)(inputs_rnn_unidades)
    rnn_unidades = layers.LSTM(input_rnn_unidades) (rnn_unidades)
    
    inputs_mejoras = keras.Input(shape=(input_mejoras,))
    rnn_mejoras = layers.Embedding(vocab_mejoras, embedding_mejoras, input_length=input_mejoras)(inputs_mejoras)
    rnn_mejoras = layers.LSTM(input_mejoras) (rnn_mejoras)
    
    y = layers.Concatenate(axis=1)([rnn_mejoras, rnn, rnn_unidades])
    
    output = layers.Dense(categorias, activation="softmax")(y)
    
    modelo = keras.Model(inputs=[inputs_rnn,inputs_rnn_unidades,inputs_mejoras], outputs=output, name="Red_recurrente")
    
    return modelo

modelo_rnn = modeloRNNPuro(input_shape_rnn,input_shape_mejoras,input_shape_rnn_unidades,max(vocabulario_mid)+1,max(vocabulario_unidades_mid)+1,max(vocabulario_mejoras_mid)+1,256,256,256,categorias)

#ruta_modelo = os.path.join(os.environ["SLURM_SUBMIT_DIR"],"/Modelo_recurrente/Input_triple_midgame")
ruta_modelo = "./Modelo_recurrente/Input_triple_midgame"
checkpoint_best = ModelCheckpoint(filepath=ruta_modelo, monitor='val_loss',verbose=1, save_best_only=True, mode='max')
lrschedule_1 = ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1, factor=0.70, mode='auto')

modelo_rnn.compile(
    loss=keras.losses.CategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(lr=lr),
    metrics=["accuracy"]
)

historia = modelo_rnn.fit(
    x=[X_train_estructuras,X_train_unidades,X_train_mejoras], y=y_train, 
    validation_data=([X_test_estructuras,X_test_unidades,X_test_mejoras], y_test), 
    batch_size=batch_size, 
    epochs=30,
    callbacks=[checkpoint_best, lrschedule_1]
)




with open('./Modelo_recurrente/historia_input_triple', 'wb') as file_pi:
    pickle.dump(historia, file_pi)
