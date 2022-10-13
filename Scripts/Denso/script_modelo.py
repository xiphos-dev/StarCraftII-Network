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

import numpy as np
import pickle
import os
import bz2


batch_size = 256

units = 64
output_size = 10 

lr = 5e-3

'''
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
'''

with open("y_train","rb") as arc:
    y_train = pickle.load(arc)

with open("y_test","rb") as arc:
    y_test = pickle.load(arc)

with open("encoder","rb") as arc:
    encoder = pickle.load(arc)

ifile = bz2.BZ2File("X_train","rb")
X_train = pickle.load(ifile)
ifile.close()

ifile = bz2.BZ2File("X_test","rb")
X_test = pickle.load(ifile)
ifile.close()

categorias = y_train.shape[1]

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

'''
numero_neuronas = [128,256,512]

for numero in numero_neuronas:

    numero_str = str(numero)
    ruta_modelo = "./Modelo_b"+numero_str
    checkpoint_best = ModelCheckpoint(filepath=ruta_modelo, monitor='val_accuracy',verbose=1, save_best_only=True, mode='auto')
    lrschedule_1 = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.70, mode='auto')


    modelo_denso = modeloDenso(X_train.shape[1],categorias,numero)

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

'''

numero_neuronas = [128,256,512]

for numero in numero_neuronas:

    numero_str = str(numero)
    ruta_modelo = "./Modelo"+numero_str+"x"+numero_str
    checkpoint_best = ModelCheckpoint(filepath=ruta_modelo, monitor='val_accuracy',verbose=1, save_best_only=True, mode='auto')
    lrschedule_1 = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.70, mode='auto')


    modelo_denso = modeloDenso2capas(X_train.shape[1],categorias,numero,numero,False)
    modelo_denso.compile(
        loss=keras.losses.CategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(lr=lr),
        metrics=["accuracy"]
    )



    historia= modelo_denso.fit(
                x=X_train, 
                y=y_train, 
                batch_size=64, 
                epochs=70, 
                verbose=True, 
                validation_data=(X_test, y_test),
                callbacks=[checkpoint_best, lrschedule_1])

    with open('./Historial_referencia/historial'+numero_str+"x"+numero_str, 'wb') as file_pi:
        pickle.dump(historia.history, file_pi)



'''
# # 128 + Dropout - Midgame prediction

# In[50]:


modelo_denso = modeloDenso(input_shape,categorias,128,True,.1)
ruta_modelo = "./Modelo128_drop"
checkpoint_best = ModelCheckpoint(filepath=ruta_modelo, monitor='val_accuracy',verbose=1, save_best_only=True, mode='auto')
lrschedule_1 = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.70, mode='auto')

modelo_denso.compile(
    loss=keras.losses.CategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(lr=lr),
    metrics=["accuracy"]
)


historia= modelo.fit(
            x=X_train, 
            y=y_train, 
            batch_size=64, 
            epochs=50, 
            verbose=True, 
            validation_data=(X_test, y_test),
            callbacks=[checkpoint_best, lrschedule_1])


with open('./Historial_referencia/historial128_drop', 'wb') as file_pi:
    pickle.dump(historia.history, file_pi)

# # 256 + Dropout - Midgame prediction

# In[51]:


modelo_denso = modeloDenso(input_shape,categorias,256,True,.1)
ruta_modelo = "./Modelo256_drop"
checkpoint_best = ModelCheckpoint(filepath=ruta_modelo, monitor='val_accuracy',verbose=1, save_best_only=True, mode='auto')
lrschedule_1 = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.70, mode='auto')

modelo_denso.compile(
    loss=keras.losses.CategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(lr=lr),
    metrics=["accuracy"]
)


historia= modelo.fit(
            x=X_train, 
            y=y_train, 
            batch_size=64, 
            epochs=50, 
            verbose=True, 
            validation_data=(X_test, y_test),
            callbacks=[checkpoint_best, lrschedule_1])


with open('./Historial_referencia/historial256_drop', 'wb') as file_pi:
    pickle.dump(historia.history, file_pi)

# In[54]:


modelo_denso = modeloDenso(input_shape,categorias,512,True,.1)
ruta_modelo = "./Modelo512_drop"
checkpoint_best = ModelCheckpoint(filepath=ruta_modelo, monitor='val_accuracy',verbose=1, save_best_only=True, mode='auto')
lrschedule_1 = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.70, mode='auto')

modelo_denso.compile(
    loss=keras.losses.CategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(lr=lr),
    metrics=["accuracy"]
)


historia= modelo.fit(
            x=X_train, 
            y=y_train, 
            batch_size=64, 
            epochs=50, 
            verbose=True, 
            validation_data=(X_test, y_test),
            callbacks=[checkpoint_best, lrschedule_1])


with open('./Historial_referencia/historial512_drop', 'wb') as file_pi:
    pickle.dump(historia.history, file_pi)

'''

'''

# # 1024 + Dropout - Midgame prediction

# In[55]:


modelo_denso = modeloDenso(input_shape,categorias,1024,True,.1)
modelo_denso.compile(
    loss=keras.losses.CategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(lr=lr),
    metrics=["accuracy"]
)


historia = modelo_denso.fit(
   	x=X_train, y=y_train,
	validation_data=(X_test, y_test),
    batch_size=batch_size, 
    epochs=15
)

graficar(historia)


# In[56]:


tf.keras.utils.plot_model(modelo_denso, show_shapes=True)


# # 512 + 512 - Midgame

# In[61]:


modelo_denso = modeloDenso2capas(input_shape,categorias,512,512,False)
modelo_denso.compile(
    loss=keras.losses.CategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(lr=lr),
    metrics=["accuracy"]
)


historia = modelo_denso.fit(
   	x=X_train, y=y_train,
	validation_data=(X_test, y_test),
    batch_size=batch_size, 
    epochs=15
)

graficar(historia)


# # 512 + 256 - Midgame

# In[60]:


modelo_denso = modeloDenso2capas(input_shape,categorias,512,256,False)
modelo_denso.compile(
    loss=keras.losses.CategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(lr=lr),
    metrics=["accuracy"]
)


historia = modelo_denso.fit(
   	x=X_train, y=y_train,
	validation_data=(X_test, y_test),
    batch_size=batch_size, 
    epochs=15
)

graficar(historia)


# # 256 + 128 - Midgame

# In[62]:


modelo_denso = modeloDenso2capas(input_shape,categorias,256,128,False)
modelo_denso.compile(
    loss=keras.losses.CategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(lr=lr),
    metrics=["accuracy"]
)


historia = modelo_denso.fit(
   	x=X_train, y=y_train,
	validation_data=(X_test, y_test),
    batch_size=batch_size, 
    epochs=15
)

graficar(historia)


# # 128 + 128 - Midgame

# In[63]:


modelo_denso = modeloDenso2capas(input_shape,categorias,128,128,False)
modelo_denso.compile(
    loss=keras.losses.CategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(lr=lr),
    metrics=["accuracy"]
)


historia = modelo_denso.fit(
   	x=X_train, y=y_train,
	validation_data=(X_test, y_test),
    batch_size=batch_size, 
    epochs=15
)

graficar(historia)


# # 256+128+128 - Midgame

# In[68]:


modelo_denso = modeloDenso3capas(input_shape,categorias,128,128,128,False)
modelo_denso.compile(
    loss=keras.losses.CategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(lr=lr),
    metrics=["accuracy"]
)


historia = modelo_denso.fit(
   	x=X_train, y=y_train,
	validation_data=(X_test, y_test),
    batch_size=batch_size, 
    epochs=15
)

graficar(historia)


# In[69]:


tf.keras.utils.plot_model(modelo_denso, show_shapes=True)


# In[ ]:





# # Estructuras 

# ## Red entrenada con p=0.8

# In[36]:


input_random = inputDensoRandomSelection(X,252,0.8)
X_train, X_test, y_train, y_test = train_test_split(input_random, dummy_y, test_size=0.20, random_state=98)

modelo_denso = modeloDenso(input_shape,categorias)
modelo_denso.compile(
    loss=keras.losses.CategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(lr=lr),
    metrics=["accuracy"]
)


modelo_denso.summary()


# In[37]:


tf.keras.utils.plot_model(modelo_denso, show_shapes=True)


# In[38]:


modelo_denso.fit(
   	x=X_train, y=y_train,
	validation_data=(X_test, y_test),
    batch_size=batch_size, 
    epochs=15
)


# ### Entrega resultados similares a p=1

# ## Red entrenada con p = 0.5

# In[39]:


input_random = inputDensoRandomSelection(X,252,0.5)
X_train, X_test, y_train, y_test = train_test_split(input_random, dummy_y, test_size=0.20, random_state=98)

modelo_denso = modeloDenso(input_shape,categorias)
modelo_denso.compile(
    loss=keras.losses.CategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(lr=lr),
    metrics=["accuracy"]
)


modelo_denso.summary()


# In[40]:


tf.keras.utils.plot_model(modelo_denso, show_shapes=True)


# In[41]:


modelo_denso.fit(
   	x=X_train, y=y_train,
	validation_data=(X_test, y_test),
    batch_size=batch_size, 
    epochs=15
)


# ## Red entrenada con p=0.2

# In[42]:


input_random = inputDensoRandomSelection(X,252,0.2)
X_train, X_test, y_train, y_test = train_test_split(input_random, dummy_y, test_size=0.20, random_state=98)

modelo_denso = modeloDenso(input_shape,categorias)
modelo_denso.compile(
    loss=keras.losses.CategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(lr=lr),
    metrics=["accuracy"]
)


modelo_denso.summary()


# In[43]:


tf.keras.utils.plot_model(modelo_denso, show_shapes=True)


# In[44]:


modelo_denso.fit(
   	x=X_train, y=y_train,
	validation_data=(X_test, y_test),
    batch_size=batch_size, 
    epochs=15
)


# # Filtrar mejoras 

# ## p = 0.8

# In[45]:


input_random = inputDensoRandomSelection(X,252,0.8,True,True)
X_train, X_test, y_train, y_test = train_test_split(input_random, dummy_y, test_size=0.20, random_state=98)

modelo_denso = modeloDenso(input_shape,categorias)
modelo_denso.compile(
    loss=keras.losses.CategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(lr=lr),
    metrics=["accuracy"]
)


modelo_denso.summary()


# In[46]:


modelo_denso.fit(
   	x=X_train, y=y_train,
	validation_data=(X_test, y_test),
    batch_size=batch_size, 
    epochs=15
)


# ## p = 0.5

# In[49]:


input_random = inputDensoRandomSelection(X,252,0.5,True,True)
X_train, X_test, y_train, y_test = train_test_split(input_random, dummy_y, test_size=0.20, random_state=98)

modelo_denso = modeloDenso(input_shape,categorias)
modelo_denso.compile(
    loss=keras.losses.CategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(lr=lr),
    metrics=["accuracy"]
)


modelo_denso.summary()


# In[50]:


modelo_denso.fit(
   	x=X_train, y=y_train,
	validation_data=(X_test, y_test),
    batch_size=batch_size, 
    epochs=15
)


# # p = 0.2 

# In[51]:


input_random = inputDensoRandomSelection(X,252,0.2,True,True)
X_train, X_test, y_train, y_test = train_test_split(input_random, dummy_y, test_size=0.20, random_state=98)

modelo_denso = modeloDenso(input_shape,categorias)
modelo_denso.compile(
    loss=keras.losses.CategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(lr=lr),
    metrics=["accuracy"]
)


modelo_denso.summary()


# In[52]:


modelo_denso.fit(
   	x=X_train, y=y_train,
	validation_data=(X_test, y_test),
    batch_size=batch_size, 
    epochs=15
)


# # Filtrado de unidades

# ## p = 0.8

# In[53]:


input_random = inputDensoRandomSelection(X,252,0.8,True,True,True)
X_train, X_test, y_train, y_test = train_test_split(input_random, dummy_y, test_size=0.20, random_state=98)

modelo_denso = modeloDenso(input_shape,categorias)
modelo_denso.compile(
    loss=keras.losses.CategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(lr=lr),
    metrics=["accuracy"]
)


modelo_denso.summary()


# In[54]:


modelo_denso.fit(
   	x=X_train, y=y_train,
	validation_data=(X_test, y_test),
    batch_size=batch_size, 
    epochs=15
)


# ## p = 0.5

# In[55]:


input_random = inputDensoRandomSelection(X,252,0.5,True,True,True)
X_train, X_test, y_train, y_test = train_test_split(input_random, dummy_y, test_size=0.20, random_state=98)

modelo_denso = modeloDenso(input_shape,categorias)
modelo_denso.compile(
    loss=keras.losses.CategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(lr=lr),
    metrics=["accuracy"]
)


modelo_denso.summary()


# In[56]:


modelo_denso.fit(
   	x=X_train, y=y_train,
	validation_data=(X_test, y_test),
    batch_size=batch_size, 
    epochs=15
)


# ## p = 0.2

# In[33]:


input_random = inputDensoRandomSelection(X,252,-1,True,True,True)
X_train, X_test, y_train, y_test = train_test_split(input_random, dummy_y, test_size=0.20, random_state=98)


# In[34]:


X_train[0]


# In[36]:


modelo_denso = modeloDenso(input_shape,categorias)
modelo_denso.compile(
    loss=keras.losses.CategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(lr=lr),
    metrics=["accuracy"]
)


modelo_denso.summary()


# In[37]:


modelo_denso.fit(
   	x=X_train, y=y_train,
	validation_data=(X_test, y_test),
    batch_size=batch_size, 
    epochs=20
)


# # Con dato clave

# In[30]:


input_random = inputDensoRandomSelectionConDatoClave(X,700,0.8,True,True,True)
modelo_denso = modeloDenso(input_shape,categorias)
modelo_denso.compile(
    loss=keras.losses.CategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(lr=lr),
    metrics=["accuracy"]
)



modelo_denso.fit(
   	x=X_train, y=y_train,
	validation_data=(X_test, y_test),
    batch_size=batch_size, 
    epochs=15
)

'''
