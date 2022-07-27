import os
import pickle
import numpy as np

with open("./X_train","rb") as arc:
    X_train = pickle.load(arc)

with open("./X_test","rb") as arc:
    X_test = pickle.load(arc)

with open("./y_train","rb") as arc:
    y_train = pickle.load(arc)

with open("./y_test","rb") as arc:
    y_test = pickle.load(arc)


categorias = y_train.shape[1]

from tensorflow.keras import layers, models


def crearRedConvolusional(forma,categorias):
    model = models.Sequential()
    model.add(layers.Conv2D(16, (3, 3), strides=1,activation='relu', data_format="channels_first", input_shape=forma))
    model.add(layers.MaxPooling2D((2, 2), strides=1))
    model.add(layers.Conv2D(32, (3, 3), strides=1, data_format="channels_first", activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='relu'))
    model.add(layers.Dense(categorias, activation="softmax"))
    return model


modelo = crearRedConvolusional(X_train.shape[1:],categorias)

y_train = np.asarray(y_train)
y_test = np.asarray(y_test)


from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras as ks


ruta = os.path.join(os.environ["SLURM_SUBMIT_DIR"],"Modelo_convolucional")
checkpoint_best = ModelCheckpoint(filepath=ruta, monitor='val_loss',verbose=1, save_best_only=True, mode='min')
lrschedule_1 = ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1, factor=0.70, mode='min')

modelo.compile(loss='categorical_crossentropy', 
                     optimizer=ks.optimizers.Adam(lr=0.003, decay=5e-4), 
                     #metrics=[tf.keras.metrics.CategoricalCrossentropy()]
                    )

historia= modelo.fit(
            x=X_train, 
            y=y_train, 
            batch_size=16, 
            epochs=50, 
            verbose=True, 
            validation_data=(X_test, y_test),
            callbacks=[checkpoint_best, lrschedule_1])


ruta = os.path.join(os.environ["SLURM_SUBMIT_DIR"],"trainHistoryDict")
with open('ruta', 'wb') as file_pi:
    pickle.dump(historia.history, file_pi)
