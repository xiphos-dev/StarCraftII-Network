#!/usr/bin/env python
# coding: utf-8

import pickle
import os
import numpy as np




with open("X_train","rb") as arc:
    X_train = pickle.load(arc)

with open("X_test","rb") as arc:
    X_test = pickle.load(arc)

with open("y_train","rb") as arc:
    y_train = pickle.load(arc)

with open("y_test","rb") as arc:
    y_test = pickle.load(arc)


with open("vocabulario","rb") as arc:
    vocabulario_mid = pickle.load(arc)

input_shape_rnn = X_train.shape[1]
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

def modeloRNNPuro_2_inputs(input_rnn,input_mejoras,vocab,vocab_mejoras,embedding_dim=256,embedding_mejoras=256,categorias=2):
    inputs_rnn = keras.Input(shape=(input_rnn,))
    rnn = layers.Embedding(vocab, embedding_dim, input_length=input_rnn)(inputs_rnn)
    rnn = layers.LSTM(input_rnn) (rnn)
    
    inputs_mejoras = keras.Input(shape=(input_mejoras,))
    rnn_mejoras = layers.Embedding(vocab_mejoras, embedding_mejoras, input_length=input_mejoras)(inputs_mejoras)
    rnn_mejoras = layers.LSTM(input_mejoras) (rnn_mejoras)
    
    y = layers.Concatenate(axis=1)([rnn_mejoras, rnn])
    
    output = layers.Dense(categorias, activation="softmax")(y)
    
    modelo = keras.Model(inputs=[inputs_rnn,inputs_mejoras], outputs=output, name="Red_recurrente")
    
    return modelo



def modeloMixto(input_rnn,input_mejoras,vocab,vocab_mejoras,embedding_dim=256,embedding_mejoras=256,categorias=2):
    inputs_cnn = keras.Input(shape=(input_cnn,))
    cnn = layers.Conv2D(16, 3, activation='relu', data_format="channels_first", input_shape=forma)(inputs_cnn)
    cnn = layers.MaxPooling2D(data_format="channels_first") (cnn)
    cnn = layers.Conv2D(32, 3, data_format="channels_first", activation='relu') (cnn)
    cnn = layers.MaxPooling2D(data_format="channels_first") (cnn)
    cnn = layers.Conv2D(16, 3, data_format="channels_first", activation='relu') (cnn)
    cnn = layers.MaxPooling2D(data_format="channels_first") (cnn)
    cnn = layers.Flatten(data_format="channels_first") (cnn)
    cnn = layers.Dense(categorias, activation="softmax")   
    cnn.load_weights("../Convolucional/Weights/F32F_filtrado_puro/")
    cnn.trainable = False
    

    inputs_rnn = keras.Input(shape=(input_rnn,))
    rnn = layers.Embedding(vocab, 100, mask_zero=True, input_length=input_shape_rnn)(inputs_mejoras)
    rnn = layers.LSTM(128) (rnn)
    rnn = layers.Dense(categorias, activation="softmax")(rnn)
    rnn.load_weights("./Weights/60lstm_filtrado_puro/")
    rnn.trainable = False

    y = layers.Concatenate(axis=1)([cnn, rnn])

    denso = layers.Dense(256, activation="softmax")(y)
    output = layers.Dense(categorias, activation="softmax")(denso)
    
    modelo = keras.Model(inputs=[inputs_cnn,inputs_rnn], outputs=output, name="Red_mixta")
    
    return modelo


def crearRedConvolucionalSinPadding(forma,categorias):
    model = models.Sequential()
    model.add(layers.Conv2D(16, 3, activation='relu', data_format="channels_first", input_shape=forma))
    model.add(layers.MaxPooling2D(data_format="channels_first"))
    model.add(layers.Conv2D(32, 3, data_format="channels_first", activation='relu'))
    model.add(layers.MaxPooling2D(data_format="channels_first"))
    model.add(layers.Conv2D(16, 3, data_format="channels_first", activation='relu'))
    model.add(layers.MaxPooling2D(data_format="channels_first"))
    model.add(layers.Flatten(data_format="channels_first"))
   # model.add(layers.Dropout(.15))
    #model.add(layers.Dense(1, activation='relu'))
    model.add(layers.Dense(512))
    model.add(layers.Dense(categorias, activation="softmax"))
    return model

def modeloRNN(num_unidades,vocab,timesteps,categorias=2):
    model = keras.Sequential()
    #model.add(layers.Embedding(vocab, 100, mask_zero=True, input_length=input_shape_rnn))

    #model.add(layers.GRU(32))
    model.add(layers.Embedding(vocab, 100, mask_zero=True, input_length=input_shape_rnn))
    model.add(layers.Masking(mask_value=0, input_shape=(timesteps,1)))
    model.add(layers.LSTM(10))
    #model.add(layers.SimpleRNN(10))

    #model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(.2))

    model.add(layers.Dense(categorias, activation="softmax"))

    model.summary()
    return model


def modeloDobleRNNDenso(input_rnn,input_denso,input_rnn_unidades,vocab,vocab_unidades,embedding_dim=256,embedding_unidades=256,categorias=2):
    inputs_rnn = keras.Input(shape=(input_rnn,))
    rnn = layers.Embedding(vocab, embedding_dim, mask_zero=True, input_length=input_rnn)(inputs_rnn)
    rnn = layers.LSTM(16) (rnn)
    
    inputs_rnn_unidades = keras.Input(shape=(input_rnn_unidades,))
    rnn_unidades = layers.Embedding(vocab_unidades, embedding_dim, mask_zero=True, input_length=input_rnn)(inputs_rnn_unidades)
    rnn_unidades = layers.LSTM(16) (rnn_unidades)
    
    inputs_denso = keras.Input(shape=(input_denso,))
    denso = layers.Dense(128, activation="relu")(inputs_denso)
    
    y = layers.Concatenate(axis=1)([denso, rnn, rnn_unidades])
    
    output = layers.Dense(categorias)(y)
    
    modelo = keras.Model(inputs=[inputs_rnn,inputs_rnn_unidades,inputs_denso], outputs=output, name="Red_recurrente_y_densa")
    
    return modelo
model = modeloRNN(X_train.shape[1],max(vocabulario_mid)+1,218,len(encoder_mid.values()))



#ruta_modelo = os.path.join(os.environ["SLURM_SUBMIT_DIR"],"/Modelo_recurrente/Input_triple_midgame")
ruta_modelo = "./Modelo_input_b_toss"
checkpoint_best = ModelCheckpoint(filepath=ruta_modelo, monitor='val_accuracy',verbose=1, save_best_only=True, mode='max')
lrschedule_1 = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.70, mode='auto')

modelo_rnn.compile(
    loss=keras.losses.CategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(lr=lr),
    metrics=["accuracy"]
)

historia = modelo_rnn.fit(
    x=X_train, y=y_train, 
    validation_data=(X_test y_test), 
    batch_size=64, 
    epochs=50,
    callbacks=[checkpoint_best, lrschedule_1]
)




with open ('./Historial_referencia/historia_input_b_toss', 'wb') as file_pi:
    pickle.dump(historia, file_pi)