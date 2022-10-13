#!/usr/bin/env python
# coding: utf-8


import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
import numpy as np


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint

import pickle
import os
import bz2


batch_size = 128

units = 64
output_size = 10 

lr = 5e-3

with open("y_train_b","rb") as arc:
    y_train = pickle.load(arc)

with open("y_test_b","rb") as arc:
    y_test = pickle.load(arc)

with open("encoder","rb") as arc:
    encoder = pickle.load(arc)

ifile = bz2.BZ2File("X_train_b","rb")
X_train = pickle.load(ifile)
ifile.close()

ifile = bz2.BZ2File("X_test_b","rb")
X_test = pickle.load(ifile)
ifile.close()

categorias = y_train.shape[1]


def modeloDenso(input_denso_shape,categorias=2,neuronas=256,drop=False,drop_prob=0):
    inputs_denso = keras.Input(shape=(input_denso_shape,))
    denso = layers.Dense(neuronas, activation="relu")(inputs_denso)
    next_layer = None
    if drop:
        dropout = layers.Dropout(drop_prob)(denso)
        next_layer = dropout
    else:
        next_layer = denso
    
    output = layers.Dense(categorias, activation="softmax")(next_layer)
    
    modelo = keras.Model(inputs=inputs_denso, outputs=output, name="Red_densa")
    
    return modelo


# In[31]:


def modeloDenso2capas(input_denso_shape,categorias=2,neuronas=256,neuronas_2=128,drop=False,drop_prob=0):
    inputs_denso = keras.Input(shape=(input_denso_shape,))
    denso = layers.Dense(neuronas, activation="relu")(inputs_denso)
    
    denso = layers.Dense(neuronas_2, activation="relu")(denso)
    
    next_layer = None
    if drop:
        dropout = layers.Dropout(drop_prob)(denso)
        next_layer = dropout
    else:
        next_layer = denso
    
    output = layers.Dense(categorias, activation="softmax")(next_layer)
    
    modelo = keras.Model(inputs=inputs_denso, outputs=output, name="Red_densa")
    
    return modelo


# In[32]:


def modeloDenso3capas(input_denso_shape,categorias=2,neuronas=256,neuronas_2=128,neuronas_3=128,drop=False,drop_prob=0):
    inputs_denso = keras.Input(shape=(input_denso_shape,))
    denso = layers.Dense(neuronas, activation="relu")(inputs_denso)
    
    denso = layers.Dense(neuronas_2, activation="relu")(denso)
    
    denso = layers.Dense(neuronas_3, activation="relu")(denso)
    next_layer = None
    if drop:
        dropout = layers.Dropout(drop_prob)(denso)
        next_layer = dropout
    else:
        next_layer = denso
    
    output = layers.Dense(categorias, activation="softmax")(next_layer)
    
    modelo = keras.Model(inputs=inputs_denso, outputs=output, name="Red_densa")
    
    return modelo

numero_neuronas = [128,256,512]

for numero in numero_neuronas:

    numero_str = str(numero)
    ruta_modelo = "./Modelo_b"+numero_str
    checkpoint_best = ModelCheckpoint(filepath=ruta_modelo, monitor='val_accuracy',verbose=1, save_best_only=True, mode='auto')
    lrschedule_1 = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.70, mode='auto')


    modelo_denso = modeloDenso(input_shape,categorias,numero)

    modelo_denso.compile(
        loss=keras.losses.CategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(lr=lr),
        metrics=["accuracy"]
    )


    historia= modelo_denso.fit(
                x=X_train, 
                y=y_train, 
                batch_size=64, 
                epochs=50, 
                verbose=True, 
                validation_data=(X_test, y_test),
                callbacks=[checkpoint_best, lrschedule_1])

    with open('./Historial_referencia/historial_b'+numero_str, 'wb') as file_pi:
        pickle.dump(historia.history, file_pi)
