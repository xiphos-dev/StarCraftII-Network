#!/usr/bin/env python
# coding: utf-8

# 

# In[1]:


import numpy as np
import pickle
import os
import bz2

'''
with open("X_train_1","rb") as arc:
    X_train = pickle.load(arc)

with open("X_test_1","rb") as arc:
    X_test = pickle.load(arc)
'''
numero = "1"

with open("y_train_"+numero,"rb") as arc:
    y_train = pickle.load(arc)

with open("y_test_"+numero,"rb") as arc:
    y_test = pickle.load(arc)

ifile = bz2.BZ2File("X_train_"+numero,"rb")
X_train = pickle.load(ifile)
ifile.close()

ifile = bz2.BZ2File("X_test_"+numero,"rb")
X_test = pickle.load(ifile)
ifile.close()

'''
ifile = bz2.BZ2File("y_train","rb")
y_train = pickle.load(ifile)
ifile.close()

ifile = bz2.BZ2File("y_test","rb")
y_test = pickle.load(ifile)
ifile.close()
'''
categorias = y_train.shape[1]

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

from tensorflow.keras import layers, models


# In[2]:

batch_size = 16

units = 64
output_size = 10 

lr = 5e-3



def crearRedConvolucional(forma,categorias):
    model = models.Sequential()
    model.add(layers.Conv2D(8, 3, padding="same",activation='relu', data_format="channels_first", input_shape=forma))
    model.add(layers.MaxPooling2D(data_format="channels_first"))
    model.add(layers.Conv2D(16, 3, padding="same",data_format="channels_first", activation='relu'))
    model.add(layers.MaxPooling2D(data_format="channels_first"))
    model.add(layers.Conv2D(8, 3, padding="same", data_format="channels_first", activation='relu'))
    model.add(layers.MaxPooling2D(data_format="channels_first"))
    model.add(layers.Flatten(data_format="channels_first"))
    model.add(layers.Dropout(.15))
    #model.add(layers.Dense(1, activation='relu'))
    model.add(layers.Dense(512))
    model.add(layers.Dense(categorias, activation="softmax"))
    return model


def crearRedConvolucionalSinPadding(forma,categorias):
    model = models.Sequential()
    model.add(layers.Conv2D(8, 3, activation='relu', data_format="channels_first", input_shape=forma))
    model.add(layers.MaxPooling2D(data_format="channels_first"))
    model.add(layers.Conv2D(16, 3, data_format="channels_first", activation='relu'))
    model.add(layers.MaxPooling2D(data_format="channels_first"))
    model.add(layers.Conv2D(8, 3, data_format="channels_first", activation='relu'))
    model.add(layers.MaxPooling2D(data_format="channels_first"))
    model.add(layers.Flatten(data_format="channels_first"))
    #model.add(layers.Dropout(.15))
    #model.add(layers.Dense(1, activation='relu'))
    model.add(layers.Dense(512))
    model.add(layers.Dense(categorias, activation="softmax"))
    return model


#modelo = crearRedConvolucional(X_train.shape[1:],categorias)
modelo = crearRedConvolucionalSinPadding(X_train.shape[1:],categorias)

#y_train = np.asarray(y_train)
#y_test = np.asarray(y_test)


from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras as ks

ruta_modelo = "./Modelo88f_no_dropout_sinpadding_t"
checkpoint_best = ModelCheckpoint(filepath=ruta_modelo, monitor='val_accuracy',verbose=1, save_best_only=True, mode='auto')
lrschedule_1 = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.70, mode='auto')

modelo.compile(loss=ks.losses.CategoricalCrossentropy(), 
                     optimizer=ks.optimizers.Adam(lr=lr, decay=5e-4), 
                     metrics=["accuracy"]
                     #metrics=[tf.keras.metrics.CategoricalCrossentropy()]
                    )

historia= modelo.fit(
            x=X_train, 
            y=y_train, 
            batch_size=64, 
            epochs=60, 
            verbose=True, 
            validation_data=(X_test, y_test),
            callbacks=[checkpoint_best, lrschedule_1])


with open('./historial88f_no_dropout_sinpadding_t'+numero, 'wb') as file_pi:
    pickle.dump(historia.history, file_pi)
